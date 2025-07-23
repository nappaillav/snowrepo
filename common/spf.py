"""
Successful Path Finder
"""
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(layer: torch.nn.modules):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        # gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight.data, gain=1)
        if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)


def ln_activ(x: torch.Tensor, activ: Callable):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)

# def ln_activ(x: torch.Tensor, activ: Callable):
#     x = F.layer_norm(activ(x), (x.shape[-1],))
#     return x

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

class Encoder(nn.Module):
    def __init__(self, state_dim:int, pixel_obs:bool, zs_dim:int, hdim:int, activ:str='elu'):
        super().__init__()
        if pixel_obs:
            self.zs = self.cnn_zs
            self.zs_cnn1 = nn.Conv2d(state_dim, 32, 3, stride=2)
            self.zs_cnn2 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn3 = nn.Conv2d(32, 32, 3, stride=2)
            self.zs_cnn4 = nn.Conv2d(32, 32, 3, stride=1)
            self.zs_lin = nn.Linear(800, zs_dim)
        else:
            self.zs = self.mlp_zs
            self.zs_mlp = BaseMLP(state_dim, zs_dim, hdim, activ)
        self.activ = getattr(F, activ)
        self.apply(weight_init)
        
    def forward(self, state:torch.Tensor):
        return self.zs(state)
        
    def cnn_zs(self, state: torch.Tensor):
        state = state/255. - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs)).reshape(state.shape[0], -1)
        return ln_activ(self.zs_lin(zs), self.activ)

    def mlp_zs(self, state: torch.Tensor):
        return ln_activ(self.zs_mlp(state), self.activ)

class WorldModel(nn.Module):
    def __init__(self, encoder_module:nn.Module, action_dim: int,
                    zs_dim: int=512, za_dim: int=256, zsa_dim: int=512, 
                    hdim: int=512, activ: str='elu', use_mb: bool=False):
        super().__init__()
        self.zs = encoder_module
        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        if use_mb:
            self.mb = nn.Linear(zsa_dim+zs_dim, zs_dim+1)
        else:
            self.done = nn.Linear(zsa_dim+zs_dim, 1)

        self.zs_dim = zs_dim
        self.use_mb = use_mb
        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))

    def model_all(self, zs: torch.Tensor, action: torch.Tensor, zg:torch.Tensor=None):
        zsa = self.forward(zs, action)
        if self.use_mb:
            zsd = self.mb(torch.cat([zsa, zg], 1))
            return zsd[:, 1:], zsd[:, 0:1] 
        done = self.done(torch.cat([zsa, zg], 1))
        return zsa, done

class WorldModelV2(nn.Module):
    def __init__(self, encoder_module:nn.Module, action_dim: int,
                    zs_dim: int=512, za_dim: int=256, zsa_dim: int=512, 
                    hdim: int=512, activ: str='elu', use_mb: bool=False, 
                    goal_dim=128, latent_dim=512):
        super().__init__()
        self.zs = encoder_module
        self.goal_rep = nn.Linear(2*zs_dim, goal_dim)
        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        
        self.mb = BaseMLP(zsa_dim, latent_dim, hdim, activ)
        self.next_state = nn.Linear(latent_dim, zs_dim)
        self.value = nn.Linear(latent_dim+goal_dim, 1)
        self.pred_action = BaseMLP(latent_dim+zs_dim, action_dim, hdim, 'tanh')
        
        self.zs_dim = zs_dim
        self.use_mb = use_mb
        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))
    
    def goal_emb(self, zs:torch.Tensor, zg:torch.Tensor):
        # Goal embedding follows Same Layer normalization
        return ln_activ(self.goal_rep(torch.cat([zs,zg], 1)), self.activ)

    def model_all(self, zs, action, next_zs, zg):
        """
        World Model:
        - Predict next_zs
        - Predict action (inverse model)
        - Predict return/success probability
        """
        zsa = self.forward(zs, action)
        latent = self.mb(zsa)

        # goal embedding
        goal_emb = self.goal_emb(zs, zg)

        # inverse model: from zs and next_zs
        pred_action = self.pred_action(torch.cat([zs, next_zs], 1)) 

        # forward model
        pred_next_zs = self.next_state(latent)

        # reward head → this is a value estimate if label = gamma^steps_to_goal
        reward_pred = F.sigmoid(self.value(torch.cat([latent, goal_emb], 1)))

        return pred_action, pred_next_zs, reward_pred


class Policy(nn.Module):
    def __init__(self, action_dim: int, discrete: bool, gumbel_tau: float=10, 
                    zs_dim: int=512, hdim: int=512, activ: str='relu', 
                    goal_dim:int=256, encoder_module:nn.Module=None):
        """
        goal_dim: This in parallel with the 
        """
        super().__init__()
        self.goal_encoder = encoder_module if encoder_module else None
        self.policy = BaseMLP(zs_dim+goal_dim, action_dim, hdim, activ)
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete
    

    def forward(self, zs: torch.Tensor, goal:torch.Tensor):
        if self.goal_encoder:
            zsg = torch.cat([zs, self.goal_encoder(goal)], 1) 
        else:
            zsg = torch.cat([zs, goal], 1) 
        pre_activ = self.policy(zsg)
        action = self.activ(pre_activ)
        return action, pre_activ

    def act(self, zs: torch.Tensor, goal:torch.Tensor):
        action, _ = self.forward(zs, goal)
        return action


class Value(nn.Module):
    def __init__(self, zsa_dim: int=512, hdim: int=512, activ: str='elu', goal_dim:int=256,
                    encoder_module:nn.Module=None, final_activ:bool=False):
        super().__init__()

        class ValueNetwork(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hdim: int=512, activ: str='elu',
                         final_activ:bool=False):
                super().__init__()
                self.q1 = BaseMLP(input_dim, hdim, hdim, activ)
                self.q2 = nn.Linear(hdim, output_dim)

                self.activ = getattr(F, activ)
                self.final_activ = getattr(F, 'sigmoid') if final_activ else None
                self.apply(weight_init)

            def forward(self, zsa: torch.Tensor):
                zsa = ln_activ(self.q1(zsa), self.activ)
                out = self.final_activ(self.q2(zsa)) if self.final_activ else self.q2(zsa)
                return out
            
        self.goal_encoder = encoder_module if encoder_module else None
        self.q1 = ValueNetwork(zsa_dim+goal_dim, 1, hdim, activ, final_activ)
        self.q2 = ValueNetwork(zsa_dim+goal_dim, 1, hdim, activ, final_activ)


    def forward(self, zsa: torch.Tensor, goal:torch.Tensor):
        if self.goal_encoder:
            zsag = torch.cat([zsa, self.goal_encoder(goal)], 1) 
        else:
            zsag = torch.cat([zsa, goal], 1)
        return torch.cat([self.q1(zsag), self.q2(zsag)], 1)

class SubGoalPolicy(nn.Module):
    def __init__(self, zs_dim: int=512, hdim: int=512, activ: str='relu', 
                    goal_dim:int=256, encoder_module:nn.Module=None):
        
        super().__init__()
        self.goal_encoder = encoder_module if encoder_module else None
        self.policy = BaseMLP(zs_dim+goal_dim, goal_dim, hdim, activ)

    def forward(self, zs: torch.Tensor, goal:torch.Tensor):
        if self.goal_encoder:
            zsg = torch.cat([zs, self.goal_encoder(goal)], 1) 
        else:
            zsg = torch.cat([zs, goal], 1) 
        subgoal = self.policy(zsg)
        return subgoal

    def act(self, zs: torch.Tensor, goal:torch.Tensor):
        subgoal = self.forward(zs, goal)
        return subgoal
    