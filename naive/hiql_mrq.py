import copy
import dataclasses
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import buffer
import models
import utils
from largebuffer import ReplayBufferBatch

@dataclasses.dataclass
class Hyperparameters:
    # Generic
    batched_buffer:bool = False
    chunks:int = 3
    batch_size: int = 256
    buffer_size: int = 1e6
    gc_negative:bool = True
    discount: float = 0.99
    target_update_freq: int = 250
    storage_cpu:bool = False

    # Exploration
    # buffer_size_before_training: int = 10e3
    # exploration_noise: float = 0.2

    # TD3
    # target_policy_noise: float = 0.2
    # noise_clip: float = 0.3

    # TD3+BC
    # lmbda: float = 0.3
    
    #IQL
    expectile:float = 0.8

    # Encoder Loss
    dyn_weight: float = 1
    reward_weight: float = 0.1
    done_weight: float = 0.1

    # Replay Buffer (LAP)
    prioritized: bool = True
    alpha: float = 0.4
    min_priority: float = 1
    enc_horizon: int = 5
    Q_horizon: int = 3

    # Encoder Model
    zs_dim: int = 512
    zsa_dim: int = 512
    za_dim: int = 256
    enc_hdim: int = 512
    enc_activ: str = 'elu'
    enc_lr: float = 1e-4
    enc_wd: float = 1e-4
    pixel_augs: bool = True
    goal_dim:int = 0

    # Value Model
    value_hdim: int = 512
    value_activ: str = 'elu'
    value_lr: float = 3e-4
    value_wd: float = 1e-4
    value_grad_clip: float = 20

    # High Policy Model
    high_policy_hdim: int = 512
    high_policy_activ: str = 'relu'
    high_policy_lr: float = 3e-4
    high_policy_wd: float = 1e-4
    high_alpha:float = 3.0
    high_stochastic:bool = True

    # Low Policy Model
    low_policy_hdim: int = 512
    low_policy_activ: str = 'relu'
    low_policy_lr: float = 3e-4
    low_policy_wd: float = 1e-4
    low_alpha:float = 3.0
    low_stochastic:bool = True
    gumbel_tau: float = 10
    pre_activ_weight: float = 1e-5

    # Reward model
    # num_bins: int = 65
    # lower: float = -10
    # upper: float = 10

    def __post_init__(self): utils.enforce_dataclass_type(self)


class Agent:
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool, discrete: bool,
        device: torch.device, history: int=1, hp: Dict={}):
        self.name = 'HIQL+MR.Q'

        self.hp = Hyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)
        self.device = device

        # if discrete: # Scale action noise since discrete actions are [0,1] and continuous actions are [-1,1].
        #     self.exploration_noise *= 0.5
        #     self.noise_clip *= 0.5
        #     self.target_policy_noise *= 0.5

        if self.hp.batched_buffer:
            print("Using a Chunk Buffer")
            self.replay_buffer = ReplayBufferBatch(self.batch_size, weight=None, gamma=self.discount, 
                                          storage_cpu=self.storage_cpu, stack=history, chunks=self.hp.chunks)

        else:
            print("Using a Normal Buffer")
            self.replay_buffer = buffer.ReplayBuffer(self.batch_size, weight=None, gamma=self.discount, 
                                          storage_cpu=self.storage_cpu, stack=history)

        # Encoder and Encoder Target (Required)
        self.encoder = models.Encoder2(obs_shape[0], action_dim, pixel_obs,
            self.zs_dim, self.za_dim, self.zsa_dim, self.enc_hdim, self.enc_activ,
            self.goal_dim).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.enc_lr, weight_decay=self.enc_wd)
        self.encoder_target = copy.deepcopy(self.encoder)

        # High Level Policy
        low_input = self.zs_dim + self.goal_dim if self.goal_dim > 0 else 2 * self.zs_dim
        out_dim = self.zs_dim if self.goal_dim == 0 else self.goal_dim
        self.high_policy = models.Actor(2*self.zs_dim, out_dim, discrete, self.gumbel_tau, self.high_policy_hdim, self.high_policy_activ,
                                            stochastic=self.high_stochastic).to(self.device)
        self.high_policy_optimizer = torch.optim.AdamW(self.high_policy.parameters(), lr=self.high_policy_lr, weight_decay=self.high_policy_wd)
        # self.high_policy_target = copy.deepcopy(self.high_policy)

        # Low Level Policy
        self.low_policy = models.Actor(low_input, action_dim, discrete, self.gumbel_tau, self.low_policy_hdim, self.low_policy_activ,
                                           stochastic=self.low_stochastic).to(self.device)
        self.low_policy_optimizer = torch.optim.AdamW(self.low_policy.parameters(), lr=self.low_policy_lr, weight_decay=self.low_policy_wd)
        # self.low_policy_target = copy.deepcopy(self.low_policy)

        self.value = models.Value(self.zs_dim, self.value_hdim, self.value_activ).to(self.device)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=self.value_lr, weight_decay=self.value_wd)
        self.value_target = copy.deepcopy(self.value)

        # Used by reward prediction
        # Maybe not required
        # self.two_hot = TwoHot(self.device, self.lower, self.upper, self.num_bins)

        # Environment properties
        self.pixel_obs = pixel_obs
        self.state_shape = obs_shape # This includes history, horizon, channels, etc.
        self.discrete = discrete
        self.action_dim = action_dim
        self.max_action = max_action

        # Tracked values
        self.reward_scale, self.target_reward_scale = 1, 1
        self.training_steps = 0
        # ADDED 
        self.latest_encoder_metric = { 'train/encoder_loss' : 0}
        self.history = history


    def select_action(self, state: np.array, goal: np.array):

        with torch.no_grad():
            if self.pixel_obs:
                state = torch.tensor(state.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)
                goal = torch.tensor(goal.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)
            else:
                state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
                goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float, device=self.device)
            zs = self.encoder.zs(state)
            zg = self.encoder.zs(goal)
            high_dist = self.high_policy(zs, zg)
            sub_goal = high_dist.sample()
            # sub_goal = sub_goal / torch.norm(sub_goal, dim=-1, keepdim=True) * torch.sqrt(torch.tensor(sub_goal.shape[-1], dtype=sub_goal.dtype, device=sub_goal.device))

            action = self.low_policy.act(zs, sub_goal).sample()            

            return int(action.argmax()) if self.discrete else action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action
    
    
    def train(self):
        metrics = {}
        # if self.replay_buffer.size <= self.buffer_size_before_training: return

        self.training_steps += 1

        if self.hp.batched_buffer and self.training_steps % 20000 == 0:
            print("Agent: Updating Dataset")
            self.update_dataset()

        if (self.training_steps-1) % self.target_update_freq == 0:
            # self.policy_target.load_state_dict(self.policy.state_dict())#
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.target_reward_scale = self.reward_scale
            # self.reward_scale = self.replay_buffer.reward_scale()

            for _ in range(self.target_update_freq):
                # state, action, next_state, reward, not_done = self.replay_buffer.sample(self.enc_horizon, include_intermediate=True)
                if self.history == 0:
                    batch = self.replay_buffer.sample(horizon=self.enc_horizon, include_intermediate=True, gc_negative=False)
                else:
                    batch = self.replay_buffer.sample_test(horizon=self.enc_horizon, include_intermediate=True, gc_negative=False)
                # state=state, action=action, next_state=next_state, goal=goal, not_done=not_done, reward=reward
                state, action, next_state, reward, not_done = batch['state'],batch['action'],batch['next_state'],batch['reward'], batch['not_done'] 
                state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self.pixel_augs)
                enc_metrics = self.train_encoder(state, action, next_state, reward, not_done, False)

                metrics.update(enc_metrics)
        # batch = self.replay_buffer.sample(self.Q_horizon, include_intermediate=False)
        if self.history == 0:
            batch = self.replay_buffer.sample(gc_negative=self.gc_negative, horizon=self.Q_horizon, include_intermediate=False)
        else:
            batch = self.replay_buffer.sample_test(gc_negative=self.gc_negative, horizon=self.Q_horizon, include_intermediate=False)

        state, action, next_state, reward, not_done, value_goal = batch['state'],batch['action'],batch['next_state'],batch['reward'], batch['not_done'], batch['value_goal'], 
        actor_goal, actor_sub_goal = batch['actor_goal'], batch['sub_goal'] 
        state, next_state = maybe_augment_state(state, next_state, self.pixel_obs, self.pixel_augs)
        reward, term_discount = multi_step_reward(reward, not_done, self.discount)

        rl_metrics = self.train_rl(state, action, next_state, reward, value_goal, actor_goal, actor_sub_goal, term_discount,
            self.reward_scale, self.target_reward_scale)

        metrics.update(rl_metrics)

        return metrics
        


    def train_encoder(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                        reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool):
        with torch.no_grad():
            encoder_target = self.encoder_target.zs(
                next_state.reshape(-1,*self.state_shape) # Combine batch and horizon
            ).reshape(state.shape[0],-1,self.zs_dim) # Separate batch and horizon
        
        pred_zs = self.encoder.zs(state[:,0])
        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.
        dyn_losses, inv_losses, return_losses = [], [], []
        for i in range(self.enc_horizon):
            # pred_d, pred_zs, pred_r = self.encoder.model_all(pred_zs, action[:,i])
            pred_zs = self.encoder.model_all(pred_zs, action[:,i])

            # Mask out states past termination.
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            # reward_loss = (self.two_hot.cross_entropy_loss(pred_r, reward[:,i]) * prev_not_done).mean()
            # done_loss = masked_mse(pred_d, 1. - not_done[:,i].reshape(-1,1), prev_not_done) if env_terminates else 0
            # encoder_loss = encoder_loss + self.dyn_weight * dyn_loss + self.reward_weight * reward_loss + self.done_weight * done_loss

            encoder_loss = encoder_loss + self.dyn_weight * dyn_loss 
            # prev_not_done = not_done[:,i].reshape(-1,1) * prev_not_done # Adjust termination mask.

            dyn_losses.append(dyn_loss.item())

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()
        metrics = {
                    'train/encoder_loss': encoder_loss.item(),
                    'train/encoder_DynLoss': np.mean(dyn_losses),
                }
        return metrics


    def train_rl(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, value_goal:torch.Tensor, actor_goal:torch.Tensor, actor_sub_goal:torch.Tensor,
        term_discount: torch.Tensor, reward_scale: float, target_reward_scale: float):

        metrics = {}
        with torch.no_grad():
            next_zs = self.encoder_target.zs(next_state)
            zg = self.encoder_target.zs(value_goal)
            zs = self.encoder_target.zs(state)

            next_v_t = self.value_target(next_zs, zg)
            q = reward + term_discount * next_v_t.min(1,keepdim=True).values
            v_t = self.value_target(zs, zg).mean(1,keepdim=True)
            
            q_val = reward + term_discount * next_v_t # No min
            # if self.gc_negative:
            #     next_v_target.clamp(-1/(1-self.discount), 0)
            # else:
            #     next_v_target.clamp(0, 1)
            adv = q - v_t

            zs = self.encoder.zs(state)
            zg_value = self.encoder.zs(value_goal)

        V = self.value(zs, zg_value)
        # value_loss = expectile_loss(adv, q_val.mean(1, keepdim=True) - V, self.expectile)
        value_loss1 = expectile_loss(adv, (q_val[:, 0] - V[:, 0]).unsqueeze(-1), self.expectile).mean()
        value_loss2 = expectile_loss(adv, (q_val[:, 1] - V[:, 1]).unsqueeze(-1), self.expectile).mean()
        value_loss = value_loss1 + value_loss2
        # value_loss = F.smooth_l1_loss(V, V_target.expand(-1,2))

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()
        metrics.update({
            'train/critic_loss': value_loss.item(),
            'train/critic_Qmean': V.mean().item(),
            'train/critic_Qmax': V.max().item(),
            'train/critic_Qmin': V.min().item(),
            # 'train/critic_Qnorm': norm.item(),            
        })

        # Low Actor Update
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
            zg_sub = self.encoder.zs(actor_sub_goal)
            
            # Goal representation detached 
            if self.goal_dim == 0:
                goal_rep = zg_sub
            else:
                goal_rep = self.encoder.goal_rep(zs, zg_sub).detach()

            nv = self.value(next_zs, zg_sub).mean(1, keepdim=True)
            v = self.value(zs, zg_sub).mean(1, keepdim=True)
            adv = nv - v
            exp_a = torch.exp(adv * self.low_alpha).clamp(max=100.0)
        
        dist = self.low_policy(zs, goal_rep)
        log_prob = dist.log_prob(action)
        actor_loss = -(exp_a * log_prob).mean()

        self.low_policy_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.low_policy_optimizer.step()

        metrics.update({
            "train/Lactor_loss": actor_loss.item(),
            "train/Lactor_adv": adv.mean().item(),
            "train/Lactor_logprob": log_prob.mean().item(),
            "train/Lactor_mse": ((dist.mean - action)**2).mean()    
        })
        
        # High Actor Update
        with torch.no_grad():
            zg_actor = self.encoder.zs(actor_goal)
            v = self.value(zs, zg_actor).mean(1, keepdim=True)
            nv = self.value(zg_sub, zg_actor).mean(1, keepdim=True)
            adv = nv - v
            exp_a = torch.exp(adv * self.high_alpha).clamp(max=100.0)

        # use low level policy with action as this 
        dist = self.high_policy(zs, zg_actor)
        log_prob = dist.log_prob(goal_rep)
        high_actor_loss = -(exp_a * log_prob).mean()

        self.high_policy_optimizer.zero_grad(set_to_none=True)
        high_actor_loss.backward()
        self.high_policy_optimizer.step()

        metrics.update({
        'train/Hactor_loss': high_actor_loss.item(),
        'train/Hactor_adv': adv.mean().item(),
        'train/Hactor_bcloss': log_prob.mean().item(),
        'train/Hactor_mse': torch.mean((dist.mean - goal_rep) ** 2).item(),
        'train/Hactor_std': torch.mean(dist.scale).item(),
        })

        # IMPLEMENT TARGET UPDATE (Very Expensive)
        
        return metrics

    def update_dataset(self):
        if self.hp.batched_buffer == True:
            self.replay_buffer.update_dataset()
    
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


class TwoHot:
    def __init__(self, device: torch.device, lower: float=-10, upper: float=10, num_bins: int=101):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1) # symexp
        self.num_bins = num_bins


    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1,-1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind+1).clamp(0, self.num_bins-1)]
        weight = (x - lower)/(upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind+1).clamp(0, self.num_bins), weight)
        return two_hot


    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)


    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)


def realign(x, discrete: bool):
    return F.one_hot(x.argmax(1), x.shape[1]).float() if discrete else x.clamp(-1,1)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction='none') * mask).mean()


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, discount: float):
    ms_reward = 0
    scale = 1
    for i in range(reward.shape[1]):
        ms_reward += scale * reward[:,i]
        scale *= discount * not_done[:,i]
    
    return ms_reward, scale


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state


# Random shift.
def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)

# Added
def expectile_loss(adv, diff, expectile):
    """Compute the expectile loss."""
    weight = torch.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)