import math

import numpy as np
import torch
import torch.nn as nn
from gametrackr.gym.core import Bot


# reproduces this JAX init: https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html
class VarianceScalingInitializer:
    def __init__(self, scale=1.0, mode="fan_avg", distribution="normal"):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def __call__(self, tensor):
        if self.mode == "fan_avg":
            fan = (tensor.size(-2) + tensor.size(-1)) / 2
        else:
            raise ValueError(
                "Invalid mode. Expected 'fan_in', 'fan_out', or 'fan_avg', but got {}".format(
                    self.mode
                )
            )

        if self.distribution == "uniform":
            limit = math.sqrt(3 * (self.scale / fan))
            initializer = torch.nn.init.uniform_
            args = (-limit, limit)
        else:
            raise ValueError(
                "Invalid distribution. Expected 'truncated_normal', 'normal', or 'uniform', but got {}".format(
                    self.distribution
                )
            )

        initializer(tensor, *args)
        tensor *= self.scale


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must have dimension at least two.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def apply_variance_scaling_init(net, scale=1.0):
    fan = (net.weight.size(-2) + net.weight.size(-1)) / 2
    init_w = math.sqrt(scale / fan)
    net.weight.data.uniform_(-init_w, init_w)
    net.bias.data.fill_(0)


def custom_init_mlp(
    sizes: list[int],
    activation: nn.Module,
    output_activation: nn.Module = nn.Identity,
    output_init_scaling: float = 1.0,
    dropout: float = None,
    layer_norm: bool = False,
):
    """Create a Multilayer Perceptron in one call with custom weight initialization

    Args:
        sizes (list of int): array of layers sizes
        activation (activation function): The activation function on internal layers
        output_activation (activation function): The activation function of the output layer. Defaults is`nn.Identity`.
        output_init_scaling: scaling for the VarianceScaling init

    Returns:
        torch.nn.Module: the MLP
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if j < len(sizes) - 2:
            fc = nn.Linear(sizes[j], sizes[j + 1])
            apply_variance_scaling_init(fc, scale=1.0)

            if layer_norm and j > 0:
                layers += [nn.LayerNorm(sizes[j], eps=1e-6), fc, act()]
            else:
                layers += [fc, act()]
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        else:
            fc = nn.Linear(sizes[j], sizes[j + 1])
            apply_variance_scaling_init(fc, scale=output_init_scaling)
            if layer_norm:
                layers += [nn.LayerNorm(sizes[j], eps=1e-6), fc, act()]
            else:
                layers += [fc, act()]
    m = nn.Sequential(*layers)
    return m

# two level policy, operating as in the original HIQL implementation
class HIQLDualPolicy(nn.Module, Bot):
    def __init__(
        self,
        obs_dim,
        hidden_sizes,
        action_dim,
        std=None,
        dropout=None,
        min_log_std=0,
        max_log_std=1,
        base_obs=None,
    ):
        nn.Module.__init__(self)
        self.high = MLPGaussianPolicy(
            obs_dim,
            hidden_sizes,
            int(obs_dim / 2),
            std,
            dropout,
            min_log_std,
            max_log_std,
            base_obs=base_obs,
            to_numpy=False,
            activation=nn.Identity,
        )
        # not using base ops for the low level policy as the goal will already be in good form (a waypoint)
        self.low = MLPGaussianPolicy(
            obs_dim,
            hidden_sizes,
            action_dim,
            std,
            dropout,
            min_log_std,
            max_log_std,
            base_obs=False,
        )

    @torch.no_grad()
    def _action(self, frame, **kwargs):
        waypoint = self.high._action(frame, **kwargs)
        # adding back obs to goal since pi_low learns pi(a|w,s) and pi_high learns pi(w-s|s,g)
        return self.low._action(
            {
                "goal": waypoint + frame["observation"],
                "observation": frame["observation"],
            },
            **kwargs
        )

    def reset(self, seed):
        self.low.reset(seed)
        self.high.reset(seed)


class MLPGaussianPolicy(nn.Module, Bot):
    def __init__(
        self,
        obs_dim,
        hidden_sizes,
        action_dim,
        std=None,
        dropout=None,
        min_log_std=0,
        max_log_std=1,
        base_obs=None,
        to_numpy=True,
        activation=nn.Identity,
    ):
        nn.Module.__init__(self)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.base_obs = torch.tensor(base_obs) if base_obs else base_obs
        self.to_numpy = to_numpy

        self.model = custom_init_mlp(
            sizes=[obs_dim] + hidden_sizes + [action_dim],
            activation=nn.ReLU,
            output_activation=activation,
            output_init_scaling=0.01,
            dropout=dropout,
        )

        self.ghost_params = torch.nn.Parameter(torch.randn(()))

        if std is None:
            self.log_std_logits = nn.Parameter(
                torch.zeros(action_dim, requires_grad=True)
            )
        else:
            self.log_std = torch.log(std).to(self.ghost_params.device)
        self.is_eval = False

    def forward(self, obs, is_deterministic=True):
        mean = self.model(obs)
        if self.std is None:
            # switching to clipping as in HIQL rather than sigmoid norm as in IQL. Does not look like it changed perfs.
            log_std = torch.clip(
                self.log_std_logits, self.max_log_std, self.min_log_std
            )  # min is max in yaml
            std = torch.exp(log_std)
        else:
            std = self.std
        action_dist = torch.distributions.Independent(
            torch.distributions.Normal(mean, std), reinterpreted_batch_ndims=1
        )
        if is_deterministic:
            return mean, action_dist
        else:
            return action_dist.sample(), action_dist

    @torch.no_grad()
    def _action(self, frame, **kwargs):
        if (not self.is_eval) and "eval" in kwargs:
            self.eval()
            self.is_eval = True
            if torch.is_tensor(self.base_obs):
                print("using base obs")
        if torch.is_tensor(self.base_obs):
            self.base_obs[:2] = frame["goal"]
            goal = self.base_obs
        else:
            goal = frame["goal"]

        action, _ = self.forward(
            torch.cat((goal, frame["observation"]), dim=0),
            is_deterministic=not kwargs["stochastic"],
        )

        if self.to_numpy:
            return np.clip(action.numpy(), -1, 1)
        else:
            return action

    def reset(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class GCMLPValue(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, layer_norm):
        nn.Module.__init__(self)
        self.model = custom_init_mlp(
            sizes=[obs_dim] + hidden_sizes + [1],
            activation=nn.GELU,
            layer_norm=layer_norm,
        )

    def forward(self, observation, goal):
        goal_obs = torch.cat((goal, observation), dim=1)
        return self.model(goal_obs)

    # here we do v-q instead of the q-v in the paper (but we adapt the expectile computation accordingly)
    def compute_loss(self, batch, q_pred, expectile):
        vf_pred = self.forward(batch)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * expectile + vf_sign * (1 - expectile)
        return (vf_weight * (vf_err**2)).mean()
