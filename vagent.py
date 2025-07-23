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
    s_enc:bool = False 
    goal_dim:int=128
    enc_value_loss:str='MSE'
    mb:bool = False # Not required

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
    policy_grad_clip: float = 20

    # Reward model
    # num_bins: int = 65
    # lower: float = -10
    # upper: float = 10

    def __post_init__(self): utils.enforce_dataclass_type(self)


class Agent:
    def __init__(self, obs_shape: tuple, action_dim: int, max_action: float, pixel_obs: bool, discrete: bool,
        device: torch.device, history: int=1, hp: Dict={}):
        self.name = 'SPF'

        self.hp = Hyperparameters(**hp)
        utils.set_instance_vars(self.hp, self)
        self.device = device

        
        self.replay_buffer = ReplayBuffer(self.batch_size, weight=None)

        # Goal representation takes current state and goal --> repr
        # Separate encoder need work 
        assert self.s_enc == False
        encoder_modules = functools.partial(models.Encoder, state_dim=obs_shape[0] * history, pixel_obs=pixel_obs, 
                                            zs_dim=self.zs_dim, hdim=self.enc_hdim) 
        
        # Same encoder
        self.encoder = models.WorldModelV2(encoder_module=encoder_modules(activ=self.enc_activ), 
                                           action_dim=action_dim, zs_dim=self.zs_dim, za_dim=self.za_dim, zsa_dim=self.zsa_dim,
                                           hdim=self.enc_hdim, goal_dim=self.goal_dim, activ=self.enc_activ, use_mb=self.mb).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.enc_lr, weight_decay=self.enc_wd)
        self.encoder_target = copy.deepcopy(self.encoder)

        self.policy = models.Policy(action_dim, discrete, self.gumbel_tau, self.zs_dim,
                self.policy_hdim, self.policy_activ, self.goal_dim,
                 encoder_modules(activ=self.enc_activ) if self.s_enc else None ).to(self.device)
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.policy_lr, weight_decay=self.policy_wd)
        self.policy_target = copy.deepcopy(self.policy)

        final_activ = True if self.value_loss_fn == 'BCE' else False
        self.value = models.Value(self.zsa_dim, self.value_hdim, self.value_activ, self.goal_dim,
                                  encoder_modules(activ=self.enc_activ) if self.s_enc else None, final_activ).to(self.device)
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

    def select_action(self, state: np.array, goal: np.array):

        with torch.no_grad():
            if self.pixel_obs:
                state = torch.tensor(state.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)
                goal = torch.tensor(goal.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)
            else:
                state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
                goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float, device=self.device)
            zs = self.encoder.zs(state)
            goal_rep = None
            if self.s_enc == False:
                zg = self.encoder.zs(goal)
                goal = self.encoder.goal_emb(zs, zg)
            else:
                goal = goal 
            action = self.policy.act(zs, goal)            

            return int(action.argmax()) if self.discrete else action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action
    
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
                batch = self.replay_buffer.sample(horizon=self.enc_horizon, include_intermediate=True, gc_negative=False)
                batch['state'], batch['next_state'] = maybe_augment_state(batch['state'], batch['next_state'], self.pixel_obs, self.pixel_augs)
                self.latest_encoder_metric = self.train_encoder(batch)
                # self.latest_encoder_metric = {'train/encoder_loss' : enc_loss / self.target_update_freq}

        metrics.update(self.latest_encoder_metric)
             

        batch = self.replay_buffer.sample(gc_negative=self.gc_negative, horizon=self.Q_horizon, include_intermediate=False)
        batch['state'], batch['next_state'] = maybe_augment_state(batch['state'], batch['next_state'], self.pixel_obs, self.pixel_augs)
        batch['n-reward'], batch['term_discount'] = multi_step_reward(batch['reward'], batch['not_done'], self.discount)
        
        # train_critic
        value_metrics = self.train_critic(batch)
        metrics.update(value_metrics)
        
        
        # train_actor
        actor_metrics = self.train_actor(batch)
        metrics.update(actor_metrics)

        return metrics


    def train_encoder(self, batch:dict):
        """
        state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
        reward: torch.Tensor, not_done: torch.Tensor, env_terminates: bool
        """
        with torch.no_grad():
            encoder_target = self.encoder_target.zs(
                batch['next_state'].reshape(-1,*self.state_shape) # Combine batch and horizon
            ).reshape(batch['next_state'].shape[0],-1,self.zs_dim) # Separate batch and horizon
        
        zg = self.encoder.zs(batch['goal']) # To avoid gradients and moving target
        pred_zs = self.encoder.zs(batch['state'][:,0])
        
        prev_not_done = 1 # In subtrajectories with termination, mask out losses after termination.
        encoder_loss = 0 # Loss is accumluated over enc_horizon.
        dyn_losses, inv_losses, return_losses = [], [], []
        for i in range(self.enc_horizon):
            pred_a, pred_zs, pred_v= self.encoder.model_all(pred_zs, batch['action'][:,i], encoder_target[:,i], zg)
            
            # Mask out states past termination.
            dyn_loss = masked_mse(pred_zs, encoder_target[:,i], prev_not_done)
            inv_loss = masked_mse(pred_a, batch['action'][:,i], prev_not_done)
            if self.enc_value_loss == 'bce':
                return_loss = masked_bce(pred_v, batch['reward'][:,i].reshape(-1,1), prev_not_done) #if env_terminates else 0
            else:
                return_loss = masked_mse(pred_v, batch['reward'][:,i].reshape(-1,1), prev_not_done) #if env_terminates else 0

            encoder_loss = encoder_loss + self.dyn_weight * dyn_loss + 0.2 * return_loss + 0.2 * inv_loss

            dyn_losses.append(dyn_loss.item())
            inv_losses.append(inv_loss.item())
            return_losses.append(return_loss.item())

            prev_not_done = batch['not_done'][:,i].reshape(-1,1) * prev_not_done # Adjust termination mask. 

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()
        metrics = {
                    'encoder_loss': encoder_loss.item(),
                    # 'dyn_loss_last': dyn_losses[-1],
                    # 'inv_loss_last': inv_losses[-1],
                    # 'return_loss_last': return_losses[-1],
                    'encoder_DynLoss': np.mean(dyn_losses),
                    'encoder_InvLoss': np.mean(inv_losses),
                    'encoder_VaLoss': np.mean(return_losses),
                }
        return metrics
    
    def train_critic(self, batch):
        """
        state: torch.Tensor, 
        action: torch.Tensor, 
        next_state: torch.Tensor, 
        value_goal:torch.Tensor,
        reward: torch.Tensor, 
        term_discount: torch.Tensor, 
        reward_scale: float, 
        target_reward_scale: float
        """    
        metrics = {}
        with torch.no_grad():
            next_zs = self.encoder_target.zs(batch['next_state'])

            goal_rep = None
            if self.s_enc == False:
                zg = self.encoder_target.zs(batch['value_goal'])
                goal = self.encoder_target.goal_emb(next_zs, zg)
            else:
                goal = batch['value_goal'] 

            V_target = self.value_target(next_zs, goal).min(1,keepdim=True).values
            V_target = batch['n-reward'] + batch['term_discount'] * V_target
            
            if self.gc_negative:
                V_target.clamp(-1/(1-self.discount), 0)
            else:
                V_target.clamp(0, 1)

            # clip 0 to 1/(1-gamma) in negative reward

            zs = self.encoder.zs(batch['state'])
            zsa = self.encoder(zs, batch['action'])

            goal_rep = None
            if self.s_enc == False:
                zg = self.encoder.zs(batch['value_goal'])
                goal = self.encoder.goal_emb(zs, zg)
            else:
                goal = batch['value_goal']  # detach the encoder goal

        # this is just exponential weights 
        # Advantage = AdvantageClip((Q - Q_target).mean(1, keepdim=True)))*baw.epsilon(advantage, percentile)* batch['weight'][:, None]
        Q = self.value(zsa, goal)
        if self.value_loss_fn == 'MSE': 
            value_loss = ((Q - V_target)**2).mean()
        elif self.value_loss_fn == 'Hubert_loss':
            value_loss = F.smooth_l1_loss(Q, V_target.expand(-1,2))
        elif self.value_loss_fn == 'BCE':
            # N-step return (Long Horizon Paper)
            value_loss = F.binary_cross_entropy(Q.mean(1, keepdim=True), V_target)

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

    def train_actor(self, batch):
        """
        state: torch.Tensor, 
        action: torch.Tensor, 
        next_state: torch.Tensor, 
        actor_goal:torch.Tensor,
        """
        metrics = {}
        with torch.no_grad():
            zs = self.encoder.zs(batch['state'])
        
            goal_rep = None
            if self.s_enc == False:
                zg = self.encoder.zs(batch['actor_goal'])
                goal = self.encoder.goal_emb(zs, zg)
            else:
                goal = batch['actor_goal'] 

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
        norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_grad_clip)
        self.policy_optimizer.step()

        #########################
        #         Actor         #
        ######################### 
        metrics.update({
                "train/actor_loss": loss.item(),
                "train/actor_policy_loss": policy_loss.item(),
                "train/actor_bc_loss": bc_loss.item(),
                "train/actor_pre_loss": pre_loss.item(),
                'train/actor_Qmean': Q_policy.mean().item(),
                'train/actor_Qmax': Q_policy.max().item(),
                'train/actor_Qmin': Q_policy.min().item(),
                'train/actor_norm': norm.item()
            })
        return metrics 


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


class BAW:
    def __init__(self, capacity=50000, percentile=80):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.percentile = percentile

    def add(self, advantage):
        """Adds a new advantage value to the buffer."""
        self.buffer.append(advantage)

    def get_Nth_percentile(self, percentile):
        """Returns the Nth percentile of stored advantage values."""
        if not self.buffer:
            return 0.0  # Default fallback if buffer is empty
        return np.percentile(self.buffer, percentile)

    def epsilon(self, advantage, percentile, emin=0.05):
        """Returns ε value based on whether advantage exceeds Nth percentile."""
        threshold = self.get_Nth_percentile(percentile)
        if advantage < threshold:
            return emin
        else:
            return 1.0
        
def AdvantageClip(advantage, M):
    return torch.exp(advantage).clamp(0,M)

def realign(x, discrete: bool):
    return F.one_hot(x.argmax(1), x.shape[1]).float() if discrete else x.clamp(-1,1)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction='none') * mask).mean()

def masked_bce(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.binary_cross_entropy(x, y, reduction='none') * mask).mean()

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


# DYNAMIC REWARD Scaling 
# AUGMENTATION Seems not working 
