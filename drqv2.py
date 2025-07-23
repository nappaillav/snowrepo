"""goal conditioned DRQ-v2 (Fixed encoder)
Model adapted with minor modification
"""

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import common.OfflineBuffer as buffer
import common.utils as utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, discount: float):
    ms_reward = 0
    scale = 1
    for i in range(reward.shape[1]):
        ms_reward += scale * reward[:,i]
        scale *= discount * not_done[:,i]
    
    return ms_reward, scale

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ELU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ELU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ELU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ELU(), nn.Flatten())

        self.apply(utils.weight_init)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64) 
            dummy = self.convnet(dummy)
            self.repr_dim = dummy.flatten(1).shape[1]

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        # h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(2*repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape))

        self.apply(utils.weight_init)

    def forward(self, obs, goal, std):
        zs = self.trunk(torch.cat([obs, goal], 1))

        mu = self.policy(zs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(2*repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape, hidden_dim),
            nn.ELU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape, hidden_dim),
            nn.ELU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, goal):
        zs = self.trunk(torch.cat([obs, goal], 1))
        h_action = torch.cat([zs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    
class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.discount = 0.99
        # replay_buffer
        self.replay_buffer = buffer.OGBuffer(256)

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, obs, goal, eval_mode=True):
        obs = torch.as_tensor(obs.copy(), device=self.device)
        obs = torch.as_tensor(goal.copy(), device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = 0.0
        dist = self.actor(obs, goal, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, goal):
        metrics = dict()

        with torch.no_grad():
            stddev = 0.3
            dist = self.actor(next_obs, goal, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, goal)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, goal)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, goal):
        metrics = dict()

        stddev = 0.3
        dist = self.actor(obs, goal, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, goal)
        Q = torch.min(Q1, Q2)

        bc_loss = -(0.3 * log_prob).mean()
        actor_loss = -Q.mean() + Q.abs().mean().detach() * bc_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_bc'] = bc_loss.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self):
        metrics = dict()

        obs, action, next_obs, goal, not_done, reward = self.replay_buffer.sample(horizon=3)
        reward, discount = multi_step_reward(reward, not_done, self.discount)
        # augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)
        # encode
        obs = self.encoder(obs)
        goal = self.encoder(goal)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, goal))

        # update actor
        metrics.update(self.update_actor(obs.detach(), goal.detach()))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics