# H-MRQ

import copy
import dataclasses
import numpy as np
from collections import deque
from typing import Dict, Optional
import functools
import numpy as np
import torch
import torch.nn.functional as F
from common.buffer import ReplayBuffer
import common.spf as models
import common.utils as utils
import common.worldmodel as worldmodel
from agent import *

"""
First Implement the HIQL 
Add Encoder module
"""

@dataclasses.dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    gc_negative:bool = True
    discount: float = 0.99
    target_update_freq: int = 250

    # Exploration
    buffer_size_before_training: int = 10e3
    exploration_noise: float = 0.2

    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.3

    # TD3+BC
    lmbda: float = 0.3

    # Encoder Loss
    dyn_weight: float = 1
    reward_weight: float = 0.1
    done_weight: float = 0.1

    # Replay Buffer (LAP)
    prioritized: bool = True
    alpha: float = 0.4
    min_priority: float = 1
    enc_horizon: int = 5
    Q_horizon: int = 1

    # Encoder Model
    zs_dim: int = 512
    zsa_dim: int = 512
    za_dim: int = 256
    enc_hdim: int = 512
    enc_activ: str = 'gelu'
    enc_lr: float = 1e-4
    enc_wd: float = 1e-4
    pixel_augs: bool = False
    s_enc:bool = True
    mb:bool = False

    # Value Model
    value_hdim: int = 512
    value_activ: str = 'gelu'
    value_lr: float = 3e-4
    value_wd: float = 1e-4
    value_grad_clip: float = 20
    value_loss_fn: str = 'MSE'

    # Policy Model
    policy_hdim: int = 512
    policy_activ: str = 'relu'
    policy_lr: float = 3e-4
    policy_wd: float = 1e-4
    gumbel_tau: float = 10
    pre_activ_weight: float = 1e-5


class Agent:
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool, discrete: bool,
        device: torch.device, history: int=1, hp: Dict={}):
        self.name = 'HSPF'

        self.hp = Hyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)
        self.device = device

        
        self.replay_buffer = ReplayBuffer(self.batch_size, weight=None)

        encoder_modules = functools.partial(models.Encoder, state_dim=obs_shape[0] * history, pixel_obs=pixel_obs, 
                                            zs_dim=self.zs_dim, hdim=self.enc_hdim) if self.s_enc else None 
        
        # Same encoder
        self.encoder = models.WorldModel(encoder_module=encoder_modules(activ=self.enc_activ), action_dim=action_dim, 
                                zs_dim=self.zs_dim, za_dim=self.za_dim, zsa_dim=self.zsa_dim,
                                hdim=self.enc_hdim, activ=self.enc_activ, use_mb=self.mb).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.enc_lr, weight_decay=self.enc_wd)
        self.encoder_target = copy.deepcopy(self.encoder)

        # Value(ZSA, goal)
        final_activ = True if self.value_loss_fn == 'BCE' else False
        self.value = models.Value(self.zsa_dim, self.value_hdim, self.value_activ,
                                  encoder_modules(activ=self.enc_activ), final_activ).to(self.device)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=self.value_lr, weight_decay=self.value_wd)
        self.value_target = copy.deepcopy(self.value)


        # Environment properties
        self.pixel_obs = pixel_obs
        self.state_shape = obs_shape # This includes history, horizon, channels, etc.
        self.discrete = discrete
        self.action_dim = action_dim
        self.max_action = max_action

        # Tracked values
        self.reward_scale, self.target_reward_scale = 1, 0
        self.training_steps = 0
        self.latest_encoder_metric = { 'train/encoder_loss' : 0}
    
    def select_action(self, state:np.array, goal:np.array):
        pass

    def update_target_weights(self, name: str):
        """
        Updates target network weights by copying from main to target.
        For example, name='policy' updates self.policy_target with self.policy.
        """
        source = getattr(self, name)
        target = getattr(self, f"{name}_target")
        target.load_state_dict(source.state_dict())

    def train(self):
        # if self.replay_buffer.size <= self.buffer_size_before_training: return
        metrics = {}
        self.training_steps += 1

        if (self.training_steps-1) % self.target_update_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            # self.target_reward_scale = self.reward_scale
            # state, action, next_state, goal, not_done, reward = self.replay_buffer.sample(horizon=self.Q_horizon, include_intermediate=False)
            # batch = self.replay_buffer.sample(horizon=self.Q_horizon, include_intermediate=False)
            # self.reward_scale = batch['reward'].abs().mean().item()
            enc_loss = 0
            for _ in range(self.target_update_freq):
                batch = self.replay_buffer.sample(horizon=self.enc_horizon, include_intermediate=True)
                batch['state'], batch['next_state'] = maybe_augment_state(batch['state'], batch['next_state'], self.pixel_obs, self.pixel_augs)
                enc_loss += self.train_encoder(batch)
                self.latest_encoder_metric = {'train/encoder_loss' : enc_loss / self.target_update_freq}
                

        batch = self.replay_buffer.sample(gc_negative=self.gc_negative, horizon=self.Q_horizon, include_intermediate=False)
        batch['state'], batch['next_state'] = maybe_augment_state(batch['state'], batch['next_state'], self.pixel_obs, self.pixel_augs)
        batch['n-reward'], batch['term_discount'] = multi_step_reward(batch['reward'], batch['not_done'], self.discount)

        # train_critic
        value_metrics = self.train_critic(batch)
        # train_actor
        actor_metrics = self.train_actor(batch)
        
        metrics.update(self.latest_encoder_metric)
        metrics.update(value_metrics)
        metrics.update(actor_metrics)

        return metrics

    def train_critic(self, batch):
        """
        state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, goal:torch.Tensor,
        reward: torch.Tensor, term_discount: torch.Tensor, reward_scale: float, target_reward_scale: float
        """    
        metrics = {}
        with torch.no_grad():
            next_zs = self.encoder_target.zs(batch['next_state'])

            noise = (torch.randn_like(batch['action']) * self.target_policy_noise).clamp(-self.noise_clip, self.noise_clip)
            goal = batch['goal'] if self.s_enc else self.encoder_target(batch['goal']) # IMPORTANT
            next_action = realign(self.policy_target.act(next_zs, goal) + noise, self.discrete) 

            next_zsa = self.encoder_target(next_zs, next_action) 
            Q_target = self.value_target(next_zsa, goal).min(1,keepdim=True).values
            Q_target = batch['n-reward'] + batch['term_discount'] * Q_target
            # clip 0 to 1/(1-gamma) in negative reward

            zs = self.encoder.zs(batch['state'])
            zsa = self.encoder(zs, batch['action'])
        
        goal = batch['goal'] if self.s_enc else self.encoder(batch['goal']).detach() # detach the encoder goal 
        # this is just exponential weights 
        # Advantage = AdvantageClip((Q - Q_target).mean(1, keepdim=True)))*baw.epsilon(advantage, percentile)* batch['weight'][:, None]
        Q = self.value(zsa, goal)

        if self.value_loss_fn == 'MSE': 
            value_loss = ((Q - Q_target)**2).mean()
        elif self.value_loss_fn == 'Hubert_loss':
            value_loss = F.smooth_l1_loss(Q, Q_target.expand(-1,2))
        elif self.value_loss_fn == 'BCE':
            # N-step return (Long Horizon Paper)
            value_loss = F.binary_cross_entropy(Q[:, 0:1], Q_target) + F.binary_cross_entropy(Q[:, 1:2], Q_target)

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()
        #########################
        #         Critic        #
        ######################### 
        metrics.update({
            'train/critic_loss': value_loss.item(),
            'train/critic_Qmean': Q.mean().item(),
            'train/critic_Qmax': Q.max().item(),
            'train/critic_Qin': Q.min().item(),
            'train/critic_Qnorm': norm.item(),            
        })
        return metrics

    def train_low_actor(self, batch):
        """
        state: torch.Tensor
        action: torch.Tensor
        next_state: torch.Tensor
        sub_goal:torch.Tensor
        """

        metrics = {}
        with torch.no_grad():
            zs = self.encoder.zs(batch['state'])
        
        goal = batch['sub_goal'] if self.s_enc else self.encoder(batch['sub_goal']).detach()

        policy_action, pre_activ = self.policy(zs, goal)
        zsa = self.encoder(zs, policy_action)
        Q_policy = self.value(zsa, goal)
        policy_loss = -Q_policy.mean() 
        pre_loss = self.pre_activ_weight * pre_activ.pow(2).mean()
        # BC Loss
        bc_loss = self.hp.lmbda * F.mse_loss(policy_action , batch['action'])
        loss = policy_loss + Q_policy.abs().mean().detach()*bc_loss + pre_loss
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()

        #########################
        #         Low Level Actor         #
        ######################### 
        metrics.update({
                'train/LowActor_loss': policy_loss.item(),
                'train/LowActor_Lactiv': pre_loss.item(),
                'train/LowActor_BCloss': bc_loss.item(),
                'train/LowActor_TotalLoss':loss.item(),
                'train/LowActor_Qmean': Q_policy.mean().item(),
                'train/LowActor_Qmax': Q_policy.max().item(),
                'train/LowActor_Qmin': Q_policy.min().item(),
                # 'train/actor_norm': norm.item()
            })
        return metrics 

    def train_high_actor(self, batch):
        """
        state: torch.Tensor
        action: torch.Tensor
        next_state: torch.Tensor
        high_goal:torch.Tensor
        predict the Subgoal representation 
        Model name : SubGoal
        """
        metrics = {}

        # HIQL
        """
        with torch.no_grad():
            zs = self.encoder_target.zs(batch['state'])
            zsa = self.encoder_target(zs, batch['action']) 
            # noise = (torch.randn_like(batch['action']) * self.target_policy_noise).clamp(-self.noise_clip, self.noise_clip)

            goal = batch['high_goal'] if self.s_enc else self.encoder_target(batch['high_goal'] ) # IMPORTANT
            Q_goal = self.value_target(zsa, goal).min(1,keepdim=True).values # (Q(s,a,g))
            
            zsg = self.encoder_target(batch['sub_goal'])
            sg_action = realign(self.policy_target.act(zsg, goal), self.discrete) 
            zsga = self.encoder_target(zsg, sg_action) 
            Q_subgoal = self.value_target(zsga, goal).min(1,keepdim=True).values
            # clip 0 to 1/(1-gamma) in negative reward

            adv = (Q_subgoal - Q_goal).mean(1, keepdim=True)
            exp_adv = torch.exp(adv).clamp(0, 100)
        
        """
        with torch.no_grad():
            zs = self.encoder.zs(batch['state'])
            zsa = self.encoder(zs, batch['action'])

            # zg = self.encoder.zs(batch['sub_goal'])
            # zga = self.encoder(zg, batch['sub_goal_action'])

        goal = batch['high_goal'] if self.s_enc else self.encoder_target(batch['high_goal']) # IMPORTANT
        
        sub_goal = self.subgoal(zs, goal)

        sg_loss = self.value(zsa, goal) - self.value(sub_goal, goal) 
        bc_loss = self.hp.lmbda * F.mse_loss(sub_goal , batch['sub_goal'])
        # this can be done if there is no model (Encoder.mb == False)
        loss = sg_loss + sg_loss.abs().mean().detach()*bc_loss 
        self.high_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.high_policy_optimizer.step()

    def save(self, save_folder: str):
        # Save models/optimizers
        models = [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]
        for k in models: torch.save(self.__dict__[k].state_dict(), f'{save_folder}/{k}.pt')

        # Save variables
        vars = ['hp', 'reward_scale', 'target_reward_scale', 'training_steps']
        var_dict = {k: self.__dict__[k] for k in vars}
        np.save(f'{save_folder}/agent_var.npy', var_dict)

        self.replay_buffer.save(save_folder)


    def load(self, save_folder: str):
        # Load models/optimizers.
        models = [
            'encoder', 'encoder_target', 'encoder_optimizer',
            'policy', 'policy_target', 'policy_optimizer',
            'value', 'value_target', 'value_optimizer'
        ]
        for k in models: self.__dict__[k].load_state_dict(torch.load(f'{save_folder}/{k}.pt', weights_only=True))

        # Load variables.
        var_dict = np.load(f'{save_folder}/agent_var.npy', allow_pickle=True).item()
        for k, v in var_dict.items(): self.__dict__[k] = v

        self.replay_buffer.load(save_folder)

