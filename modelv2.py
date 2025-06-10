# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(layer: torch.nn.modules):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        # nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        # if layer.bias is not None:
        #     nn.init.uniform_(layer.bias, -3e-3, 3e-3)
        # gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(layer.weight.data, gain=1)
        if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)

def small_final_init(layer):
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -3e-3, 3e-3)

def ln_activ(x: torch.Tensor, activ: Callable):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str='elu', final_init=None):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)

        self.activ = getattr(F, activ)
        self.apply(weight_init)

        if final_init is not None:
            final_init(self.l3)

    def forward(self, x: torch.Tensor):
        y = ln_activ(self.l1(x), self.activ)
        y = ln_activ(self.l2(y), self.activ)
        return self.l3(y)


class Encoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool,
        num_bins: int=65, zs_dim: int=512, za_dim: int=256, zsa_dim: int=512, hdim: int=512, activ: str='elu'):
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

        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ)
        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)

        self.zs_dim = zs_dim

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))


    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        return zsa
        # dzr = self.model(zsa)
        # return dzr[:,0:1], dzr[:,1:self.zs_dim+1], dzr[:,self.zs_dim+1:] # done, zs, reward


    def cnn_zs(self, state: torch.Tensor):
        state = state/255. - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs)).reshape(state.shape[0], -1)
        return ln_activ(self.zs_lin(zs), self.activ)

    def mlp_zs(self, state: torch.Tensor):
        return ln_activ(self.zs_mlp(state), self.activ)


class Policy(nn.Module):
    def __init__(self, action_dim: int, discrete: bool, gumbel_tau: float=10, zs_dim: int=512, hdim: int=512, activ: str='relu'):
        super().__init__()
        self.policy = BaseMLP(zs_dim, action_dim, hdim, activ, final_init=small_final_init)
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete


    def forward(self, zs: torch.Tensor):
        pre_activ = self.policy(zs)
        action = self.activ(pre_activ)
        return action, pre_activ


    def act(self, zs: torch.Tensor):
        action, _ = self.forward(zs)
        return action

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str='elu', final_init=None):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)

        self.activ = getattr(F, activ)
        self.apply(weight_init)

        if final_init is not None:
            final_init(self.l3)

    def forward(self, x: torch.Tensor):
        y = self.activ(self.l1(x))
        y = self.activ(self.l2(y))
        return F.tanh(self.l3(y))
    
class GCPolicy(nn.Module):
    def __init__(self, action_dim: int, discrete: bool, gumbel_tau: float=10, zs_dim: int=512, hdim: int=512, activ: str='relu'):
        super().__init__()
        self.policy_mean = BaseMLP(2*zs_dim, action_dim, hdim, activ, final_init=None)
        self.policy_mean = BaseMLP(2*zs_dim, action_dim, hdim, activ, final_init=None)
        # self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete
        self.activ = getattr(F, activ)

    def forward(self, zs: torch.Tensor, gzs:torch.Tensor):
        pre_activ = self.policy(torch.cat([zs, gzs], 1))
        action = self.activ(pre_activ)
        return action, pre_activ


    def act(self, zs: torch.Tensor, gzs:torch.Tensor):
        action, _ = self.forward(zs, gzs)
        return action

class Value(nn.Module):
    def __init__(self, zsa_dim: int=512, hdim: int=512, activ: str='elu'):
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

        self.q1 = ValueNetwork(zsa_dim, 1, hdim, activ)
        self.q2 = ValueNetwork(zsa_dim, 1, hdim, activ)


    def forward(self, zsa: torch.Tensor):
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)


class GCValue(nn.Module):
    def __init__(self, zsa_dim: int=512, hdim: int=512, activ: str='elu'):
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

        self.q1 = ValueNetwork(2*zsa_dim, 1, hdim, activ)
        self.q2 = ValueNetwork(2*zsa_dim, 1, hdim, activ)


    def forward(self, zsa: torch.Tensor, gz: torch.Tensor):
        return torch.cat([self.q1(torch.cat([zsa, gz], 1)), self.q2(torch.cat([zsa, gz], 1))], 1)