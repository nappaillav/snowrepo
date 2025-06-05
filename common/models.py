from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.encoder import encoder_modules

def weight_init(layer: torch.nn.modules):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        # gain = nn.init.calculate_gain('silu')
        nn.init.xavier_uniform_(layer.weight.data, gain=1.0) 
        if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)


def ln_activ(x: torch.Tensor, activ: Callable):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str='elu'):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, x: torch.Tensor):
        y = ln_activ(self.l1(x), self.activ)
        y = ln_activ(self.l2(y), self.activ)
        return self.l3(y)

class SimpleCNN(nn.Module):
    def __init__(self, state_dim, zs_dim, activ='elu'):
        super().__init__()
        self.zs_cnn = nn.Sequential(
                    nn.Conv2d(state_dim, 32, 3, stride=2),
                    nn.ELU(),
                    nn.Conv2d(32, 32, 3, stride=2),
                    nn.ELU(),
                    nn.Conv2d(32, 32, 3, stride=2),
                    nn.ELU(),
                    nn.Conv2d(32, 32, 3, stride=1),
                    nn.ELU(),
                )
        self.activ = getattr(F, activ)
        
        with torch.no_grad():
            dummy = torch.zeros(1, state_dim, 64, 64)
            dummy = self.zs_cnn(dummy)
            flat_size = dummy.flatten(1).shape[1]
        self.zs_lin = nn.Linear(flat_size, zs_dim)
    def forward(self, state):
        state = state/255. - 0.5
        zs = self.zs_lin(self.zs_cnn(state).reshape(state.shape[0], -1))
        return zs
    
class StateEncoder(nn.Module):
    def __init__(self, state_dim, pixel_obs, zs_dim, hdim, activ='elu', simple=True):
        super().__init__()
        self.pixel_obs = pixel_obs
        if pixel_obs:
            if simple:
                self.zs_cnn = SimpleCNN(state_dim, zs_dim)
            else:
                self.zs_cnn = encoder_modules['impala_small'](inp_channel=state_dim, hdim=zs_dim, layer_norm=False)
        else:
            self.zs_mlp = BaseMLP(state_dim, zs_dim, hdim, activ)

        self.activ = getattr(F, activ)
        

    def forward(self, state: torch.Tensor):        
        if self.pixel_obs:
            zs = self.zs_cnn(state)
            return ln_activ(zs, self.activ)
        else:
            return ln_activ(self.zs_mlp(state), self.activ)

class Encoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool,
        num_bins: int=65, zs_dim: int=512, za_dim: int=256, zsa_dim: int=512, hdim: int=512, activ: str='elu'):
        super().__init__()
        self.zs = StateEncoder(state_dim, pixel_obs, zs_dim, hdim, activ, simple=True)
        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        # self.model = nn.Linear(zsa_dim, zs_dim)

        self.zs_dim = zs_dim

        self.activ = getattr(F, activ)
        self.apply(weight_init)

    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))

    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        # zsa = self.model(zsa)
        return zsa # only zs 
    
    # def cnn_zs(self, state: torch.Tensor):
    #     state = state/255. - 0.5
    #     zs = self.activ(self.zs_cnn1(state))
    #     zs = self.activ(self.zs_cnn2(zs))
    #     zs = self.activ(self.zs_cnn3(zs))
    #     zs = self.activ(self.zs_cnn4(zs)).reshape(state.shape[0], -1)
    #     return ln_activ(self.zs_lin(zs), self.activ)


    # def mlp_zs(self, state: torch.Tensor):
    #     return ln_activ(self.zs_mlp(state), self.activ)


class Policy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool, discrete: bool, gumbel_tau: float=10, zs_dim: int=512, 
                 hdim: int=512, activ: str='relu', goal_encoder:bool=True):
        super().__init__()
        inp_dim = zs_dim
        if goal_encoder:
            self.gencoder = StateEncoder(state_dim, pixel_obs, zs_dim, hdim) # same activation as Policy will not be used for encoder
            inp_dim += zs_dim

        self.policy = BaseMLP(inp_dim, action_dim, hdim, activ)
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete


    def forward(self, zs: torch.Tensor, goal:torch.Tensor=None):
        if goal is not None:
            goal = self.gencoder(goal)
            pre_activ = self.policy(torch.cat([zs, goal], 1))
        else:
            pre_activ = self.policy(zs)
        action = self.activ(pre_activ)
        return action, pre_activ

    def act(self, zs: torch.Tensor, goal:torch.Tensor=None):
        action, _ = self.forward(zs, goal)
        return action


class Value(nn.Module):
    def __init__(self, state_dim:int, pixel_obs:bool, zsa_dim: int=512, hdim: int=512, activ: str='elu', 
                 goal_encoder:bool=True):
        super().__init__()

        class ValueNetwork(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hdim: int=512, activ: str='elu'):
                super().__init__()
                self.q1 = BaseMLP(input_dim, hdim, hdim, activ)
                self.q2 = nn.Linear(hdim, output_dim)

                self.activ = getattr(F, activ)
                self.apply(weight_init)

            def forward(self, zsa: torch.Tensor):
                zsa = ln_activ(self.q1(zsa), self.activ)
                return self.q2(zsa)
        
        inp_dim = zsa_dim    
        if goal_encoder:
            # Goal encoder is shared 
            self.goal_encoder = StateEncoder(state_dim, pixel_obs, zsa_dim, hdim) 
            inp_dim += zsa_dim

        self.q1 = ValueNetwork(inp_dim, 1, hdim, activ)

        self.q2 = ValueNetwork(inp_dim, 1, hdim, activ)


    def forward(self, zsa: torch.Tensor, goal:torch.Tensor=None):
        if goal is not None:
            zsa = torch.cat([zsa, self.goal_encoder(goal)], 1)
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)

class GCPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool, discrete: bool, gumbel_tau: float=10, zs_dim: int=512, 
                 hdim: int=512, activ: str='relu', goal_encoder:bool=True):
        super().__init__()
        inp_dim = zs_dim
        if goal_encoder:
            inp_dim += zs_dim

        self.policy = BaseMLP(inp_dim, action_dim, hdim, activ)
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete


    def forward(self, zs: torch.Tensor, goal:torch.Tensor=None):
        """
        Assumption here is the goal is also encoder
        """
        if goal is not None:
            pre_activ = self.policy(torch.cat([zs, goal], 1))
        else:
            pre_activ = self.policy(zs)
        action = self.activ(pre_activ)
        return action, pre_activ

    def act(self, zs: torch.Tensor, goal:torch.Tensor=None):
        action, _ = self.forward(zs, goal)
        return action


class GCValue(nn.Module):
    def __init__(self, state_dim:int, pixel_obs:bool, zsa_dim: int=512, hdim: int=512, activ: str='elu', 
                 goal_encoder:bool=True):
        super().__init__()

        class ValueNetwork(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hdim: int=512, activ: str='elu'):
                super().__init__()
                self.q1 = BaseMLP(input_dim, hdim, hdim, activ)
                self.q2 = nn.Linear(hdim, output_dim)

                self.activ = getattr(F, activ)
                self.apply(weight_init)

            def forward(self, zsa: torch.Tensor):
                zsa = ln_activ(self.q1(zsa), self.activ)
                return self.q2(zsa)
        
        inp_dim = zsa_dim    
        if goal_encoder:
            inp_dim += zsa_dim

        self.q1 = ValueNetwork(inp_dim, 1, hdim, activ)

        self.q2 = ValueNetwork(inp_dim, 1, hdim, activ)


    def forward(self, zsa: torch.Tensor, goal:torch.Tensor=None):
        if goal is not None:
            zsa = torch.cat([zsa, goal], 1)
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)

# if __name__ == '__main__':
#     # StateEncoder Test
#     encoder = StateEncoder(state_dim=3, pixel_obs=True, zs_dim=512, hdim=512, simple=False)
#     image_input = torch.randn(4, 3, 64, 64)  # Batch of 4 RGB images
#     output = encoder(image_input)
#     print(f"Total Number of Parameter : {sum(p.numel() for p in encoder.parameters())}")
#     print(encoder)
#     assert output.shape == (4, 512)

    # # Full Pipeline Test
    # encoder_net = Encoder(state_dim=3, action_dim=5, pixel_obs=True)
    # action = torch.randn(4, 5)
    # zs = encoder_net.zs(image_input)
    # zsa = encoder_net(zs, action)
    # assert zsa.shape == (4, 512)

    # # StateEncoder Test
    # encoder = StateEncoder(state_dim=10, pixel_obs=False, zs_dim=256, hdim=128)
    # state_input = torch.randn(4, 10)
    # output = encoder(state_input)
    # assert output.shape == (4, 256)

    # # Policy Test
    # policy = Policy(state_dim=10, action_dim=5, pixel_obs=False, discrete=False, goal_encoder=False)
    # zs = torch.randn(4, 512)
    # action = policy.act(zs)
    # assert action.shape == (4, 5)

    # # With Goal
    # goal_image = torch.randn(4, 3, 64, 64)
    # policy = Policy(state_dim=3, action_dim=5, pixel_obs=True, discrete=False, goal_encoder=True)
    # action = policy.act(zs, goal_image)

    # # Without Goal
    # policy = Policy(state_dim=10, action_dim=5, pixel_obs=False, discrete=True, goal_encoder=False)
    # action = policy.act(zs)  # Should not error

    # # With Goal
    # value_net = Value(state_dim=3, pixel_obs=True, goal_encoder=True)
    # zsa = torch.randn(4, 512)
    # goal_img = torch.randn(4, 3, 64, 64)
    # values = value_net(zsa, goal_img)
    # assert values.shape == (4, 2)

    # # Without Goal
    # value_net = Value(state_dim=10, pixel_obs=False, goal_encoder=False)
    # values = value_net(zsa)
    # assert values.shape == (4, 2)



