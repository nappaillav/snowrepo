import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial

def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i,p in enumerate(m):
            if p.dim() == 3: # Linear
                nn.init.trunc_normal_(p, std=0.02) # Weight
                nn.init.constant_(m[i+1], 0)
    # CNN weight initialization

# def weight_init(layer: torch.nn.modules):
#     if isinstance(layer, (nn.Linear, nn.Conv2d)):
#         # gain = nn.init.calculate_gain('silu')
#         nn.init.xavier_uniform_(layer.weight.data, gain=1.0) 
#         if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)
                
class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad
        self.padding = tuple([self.pad] * 4)

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        x = F.pad(x, self.padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim=8):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0., act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}, "\
            f"act={self.act.__class__.__name__})"


def MLP(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


def SimpleCNN(in_shape, zs_dim, num_channels=32, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    # assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(), 
        PixelPreprocess(),
        nn.Conv2d(in_shape, num_channels, 7, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten(),
        NormedLinear(512, zs_dim, act=act)]
    
    return nn.Sequential(*layers)

def ZS(state_dim, pixel_obs, zs_dim, hdim, simnorm_dim=8):
    num_channel = 32
    if pixel_obs:
        return SimpleCNN(state_dim, zs_dim, num_channels=num_channel, act=SimNorm(simnorm_dim))
    else:
        return MLP(state_dim, [hdim], zs_dim, act=SimNorm(simnorm_dim))


class Encoder(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Adapted for 
    1. TD7: https://arxiv.org/abs/2306.02451 
    2. MR.Q: https://arxiv.org/abs/2501.16142 
    
    """

    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool,
        num_bins: int=65, zs_dim: int=512, za_dim: int=256, zsa_dim: int=512, hdim: int=512, activ: str='elu'):
        super().__init__()
        simnorm_dim = int(zsa_dim // 32)
        
        self._encoder = ZS(state_dim, pixel_obs, zs_dim, hdim, simnorm_dim)
        self.za = nn.Linear(action_dim, za_dim)
        self.zs = self.encode
        self.zsa = MLP(zs_dim + za_dim , [hdim], zsa_dim, act=SimNorm(simnorm_dim))
        self.apply(weight_init)
        
    # def __repr__(self):
    #     repr = 'TD-MPC2 World Model\n'
    #     modules = ['Encoder', 'Dynamics', ]
    #     for i, m in enumerate([self._encoder, self.zsa]):
    #         if m == self._termination and not self.cfg.episodic:
    #             continue
    #         repr += f"{modules[i]}: {m}\n"
    #     repr += "Learnable parameters: {:,}".format(self.total_params)
    #     return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def encode(self, state: torch.Tensor):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        return self._encoder(state)

    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        """
        Predicts the next latent state given the current latent state and action.
        """
        za = self.za(action)
        z = torch.cat([zs, za], dim=1)
        return self.zsa(z)
    
    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        # no additional layer as we are not modelling Reward and Done 
        return zsa


class Policy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool, discrete: bool, gumbel_tau: float=10, zs_dim: int=512, 
                 hdim: int=512, activ: str='relu', goal_encoder:bool=True):
        super().__init__()
        inp_dim = zs_dim
        if goal_encoder:
            self.gencoder = ZS(state_dim, pixel_obs, zs_dim, hdim) # same activation as Policy will not be used for encoder
            inp_dim += zs_dim

        self.policy = MLP(inp_dim, [hdim], action_dim)
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
                self.q = MLP(input_dim, 2*[hdim], output_dim, dropout=0.01)

                self.apply(weight_init)

            def forward(self, zsa: torch.Tensor):
                return self.q(zsa)
        
        inp_dim = zsa_dim    
        if goal_encoder:
            # Goal encoder is shared 
            self.goal_encoder = ZS(state_dim, pixel_obs, zsa_dim, hdim) 
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

        self.policy = MLP(inp_dim, [hdim], action_dim)
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete


    def forward(self, zs: torch.Tensor, goal:torch.Tensor=None):
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
                self.q = MLP(input_dim, 2*[hdim], output_dim, dropout=0.01)

                self.apply(weight_init)

            def forward(self, zsa: torch.Tensor):
                return self.q(zsa)
        
        inp_dim = zsa_dim    
        if goal_encoder:
            # Goal encoder is shared 
            inp_dim += zsa_dim

        self.q1 = ValueNetwork(inp_dim, 1, hdim, activ)

        self.q2 = ValueNetwork(inp_dim, 1, hdim, activ)


    def forward(self, zsa: torch.Tensor, goal:torch.Tensor=None):
        if goal is not None:
            zsa = torch.cat([zsa, goal], 1)
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)