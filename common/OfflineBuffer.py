try:
    from common.ogb_utils import load_dataset
except:
    from ogb_utils import load_dataset
import numpy as np
import torch

def toTensor(array, d_type=torch.float, device=torch.device('cuda')):
    return torch.tensor(array, dtype=d_type, device=device)

def toDevice(tensor):
    if not tensor.is_cuda:
        return  tensor.to(torch.device('cuda'))
    return tensor

class OGBuffer():
    """
    Specifically for Offline Data with goal conditioning 
    """
    def __init__(self, batch_size, weight=None):
        super(OGBuffer, self).__init__()
        self.batch_size = batch_size
        self.weight = weight
    
    def load_ogbench(self, dataset_path, device='cuda'):
       
        if "visual" in dataset_path:
            self.obs_type = np.uint8
            self.normalize = 255.0
            self.pixel_obs = True
        else:
            self.obs_type = np.float32
            self.normalize = 1.0
            self.pixel_obs = False
        
        self.device = device

        dataset = load_dataset(dataset_path, self.obs_type, compact_dataset=True)
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.not_done = dataset['valids']
        self.max_size = self.state.shape[0]
        self.state_shape = self.state.shape[1:]
        self.action_shape = self.action.shape[1:]
        self.storage_device = True if self.state.is_cuda else False
        
        self.traj_length = int(torch.diff(torch.where(self.not_done == 0)[0]).float().mean())
        self.num_traj = int(self.not_done.shape[0] // self.traj_length )
        self.size = np.prod(dataset['terminals'].shape) 


    def sample(self, 
        gc_negative=True, 
        horizon=1, 
        include_intermediate=False,
        geom_p=0.99, 
        goal_p=0.2):

        batch_size = self.batch_size
        # sample index
        tid = np.random.randint(0, self.num_traj, batch_size).reshape(-1, 1)
        max_start = self.traj_length - horizon - 1 
        tpos = np.random.randint(1, max_start + 1, batch_size)
        
        # Generate horizon indices
        local_ind = tpos[:, None] + np.arange(horizon + 1)  # to handle the next state-> 
        local_ind = np.clip(local_ind, 0, self.traj_length - 1) # most not required because its done with traj-horizon
        ind = self.traj_length * tid + local_ind
        action = self.action[ind]
        
        

        if include_intermediate:
            # THIS is for Learning an encoder (dynamics model)
            both_state = self.state[ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            state = both_state[:,:-1]       # State: (batch_size, horizon, *state_dim)
            next_state = both_state[:,1:]   # Next state: (batch_size, horizon, *state_dim)
            action = action[:,:-1, :]         # Action: (batch_size, horizon, action_dim)
            if self.storage_device:
                return state, action, next_state, None, None
            else:
                return toDevice(state), toDevice(action), toDevice(next_state), None, None

        else:
            # Sample offset position with 20 as goal
            offset = np.random.geometric(p=1 - geom_p, size=batch_size) * np.where(np.random.rand(batch_size)<0.2, 0, 1)
            goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 1) # goal from 

            # Weight calculation
            if self.weight == 'exp':
                weight = 0.99 ** (goal_pos - ind[:, 0])
            elif self.weight == 'linear':
                weight = 1 - ((goal_pos - ind[:, 0]) / (self.traj_length + 1e-8))
            else:
                weight = np.ones_like(goal_pos, dtype=np.float32)
            assert np.all(weight >= 0), "Negative weights detected!"
            
            not_done = toTensor(np.where(ind > goal_pos[:, None], 0, 1)[:, :horizon]) # not done
            reward = toTensor(np.where(ind[:, :-1] == goal_pos[:, None], 0, -1)[:, :horizon])
            
            stacked_ind = np.stack((ind[:, 0], ind[:, -1], goal_pos), 1)
            all_state = self.state[stacked_ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            state = all_state[:,0]       # State: (batch_size, *state_dim)
            next_state = all_state[:,1]   # Next state: (batch_size, *state_dim)
            action = action[:,0,:]
            goal = all_state[:,2]
            if self.storage_device:
                return state, action, next_state, goal, not_done.unsqueeze(-1), reward.unsqueeze(-1)
            else:
                return toDevice(state), toDevice(action), toDevice(next_state), toDevice(goal), not_done.unsqueeze(-1), reward.unsqueeze(-1)
    

class OGBuffer2():
    """
    Specifically for Offline Data with goal conditioning 
    """
    config ={
            'value_p_curgoal': 0.3,
            'value_p_trajgoal': 0.5,
            'value_p_randomgoal': 0.2,
            'value_geom_sample':True,
            'actor_p_curgoal': 0.0,
            'actor_p_trajgoal': 1.0,
            'actor_p_randomgoal': 0.0,
            'actor_geom_sample':False
        }
    def __init__(self, batch_size, weight=None):
        super(OGBuffer2, self).__init__()
        self.batch_size = batch_size
        self.weight = weight

    
    def load_ogbench(self, dataset_path, device='cuda'):
       
        if "visual" in dataset_path:
            self.obs_type = np.uint8
            self.normalize = 255.0
            self.pixel_obs = True
        else:
            self.obs_type = np.float32
            self.normalize = 1.0
            self.pixel_obs = False
        
        self.device = device

        dataset = load_dataset(dataset_path, self.obs_type, compact_dataset=True)
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.not_done = dataset['valids']
        self.max_size = self.state.shape[0]
        self.state_shape = self.state.shape[1:]
        self.action_shape = self.action.shape[1:]
        self.storage_device = True if self.state.is_cuda else False
        
        self.traj_length = int(np.diff(np.where(self.not_done == 0)[0]).mean())
        self.num_traj = int(self.not_done.shape[0] // self.traj_length )
        self.size = np.prod(dataset['terminals'].shape)

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(dataset['terminals'] > 0)
        # self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        # assert self.terminal_locs[-1] == self.size - 1
        (self.valid_idxs,) = np.nonzero(dataset['valids'] > 0) # valid index 
    
    def sample(self, 
        gc_negative=True, 
        horizon=1, 
        include_intermediate=False,
        geom_p=0.99, 
        goal_p=0.2):

        batch_size = self.batch_size
        # sample index
        # tid = np.random.randint(0, self.num_traj, batch_size).reshape(-1, 1)
        tid = np.random.randint(0, 5, batch_size).reshape(-1, 1)
        max_start = self.traj_length - horizon - 1 
        tpos = np.random.randint(1, max_start + 1, batch_size)
        
        # Generate horizon indices
        local_ind = tpos[:, None] + np.arange(horizon + 1)  # to handle the next state-> 
        local_ind = np.clip(local_ind, 0, self.traj_length - 1) # most not required because its done with traj-horizon
        ind = self.traj_length * tid + local_ind
        action = self.action[ind]
        
        

        if include_intermediate:
            # THIS is for Learning an encoder (dynamics model)
            both_state = self.state[ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            state = both_state[:,:-1]       # State: (batch_size, horizon, *state_dim)
            next_state = both_state[:,1:]   # Next state: (batch_size, horizon, *state_dim)
            action = action[:,:-1, :]         # Action: (batch_size, horizon, action_dim)
            if self.storage_device:
                return state, action, next_state, None, None
            else:
                return toDevice(state), toDevice(action), toDevice(next_state), None, None

        else:
            value_goal_idxs = self.sample_goals(
                local_ind,
                tid,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                self.config['value_geom_sample'],
            )
            actor_goal_idxs = self.sample_goals(
                    local_ind,
                    tid,
                    self.config['actor_p_curgoal'],
                    self.config['actor_p_trajgoal'],
                    self.config['actor_p_randomgoal'],
                    self.config['actor_geom_sample'],
                )
            # Weight calculation
            # actor_weight = self._weight(actor_goal_idxs, ind)
            # value_weight = self._weight(value_goal_idxs, ind)

            successes = (ind[:, :horizon] == value_goal_idxs[:, None]).astype(float)
            reward = successes - (1.0 if gc_negative else 0.0)
            not_done = 1 -successes
            not_done = toTensor(not_done) # not done
            reward = toTensor(reward)
            
            stacked_ind = np.stack((ind[:, 0], ind[:, -1], value_goal_idxs, actor_goal_idxs), 1)
            all_state = self.state[stacked_ind].reshape(self.batch_size,-1,*self.state_shape).type(torch.float)
            state = all_state[:,0]       # State: (batch_size, *state_dim)
            next_state = all_state[:,1]   # Next state: (batch_size, *state_dim)
            action = action[:,0,:]
            value_goal = all_state[:,2]
            actor_goal = all_state[:,3]
            if self.storage_device:
                return state, action, next_state, value_goal, actor_goal, not_done, reward
            else:
                return toDevice(state), toDevice(action), toDevice(next_state), toDevice(value_goal), toDevice(actor_goal), not_done, reward
    
    def _weight(self, goal_pos, ind):
        # Weight calculation
        if self.weight == 'exp':
            weight = 0.99 ** (goal_pos - ind[:, 0])
        elif self.weight == 'linear':
            weight = 1 - ((goal_pos - ind[:, 0]) / (self.traj_length + 1e-8))
        else:
            weight = np.ones_like(goal_pos, dtype=np.float32)
        assert np.all(weight >= 0), "Negative weights detected!"
        return weight
    def random_idx(self):
        return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=self.batch_size)]
    
    def sample_goals(self, local_ind, tid, p_curgoal, p_trajgoal,  p_randomgoal, geom_sample):
        random_goal_idxs = self.random_idx()
        current_pos = self.traj_length * tid.reshape(-1) + local_ind[:,0]
        if geom_sample:
            geom_p = 0.99
            offset = np.random.geometric(p=1 - geom_p, size=self.batch_size) #* np.where(np.random.rand(batch_size)<0.2, 0, 1)
            goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(local_ind[:, 0] + offset, self.traj_length - 1)
        else:
            distances = np.random.rand(self.batch_size) #* np.where(np.random.rand(batch_size)<0.2, 0, 1)
            pos = np.round((local_ind[:, 0] * distances + (self.traj_length - 1) * (1 - distances))).astype(int)
            goal_pos = self.traj_length * tid.reshape(-1) + np.minimum(pos, self.traj_length - 1)
        
        active = np.random.rand(self.batch_size)
        goal_idxs = np.where(active < (p_trajgoal / (1.0 - p_curgoal + 1e-6)), goal_pos, random_goal_idxs)
        # Goals at the current state.
        active = np.random.rand(self.batch_size)
        goal_idxs = np.where(active < p_curgoal, current_pos, goal_idxs)
        return goal_idxs
    
# if __name__ == "__main__":
#     # dataset_path = 'F:/workspace/sai/data/visual-humanoidmaze-medium-navigate-v0-val.npz'
#     dataset_path='F:/workspace/sai/data/antmaze-medium-stitch-v0.npz'
#     buffer = OGBuffer2(4, None)
#     buffer.load_ogbench(dataset_path=dataset_path)
#     # out = buffer.sample(horizon=5, include_intermediate=True)
#     # out = buffer.sample(horizon=3, include_intermediate=False)
#     out = buffer.sample(horizon=3, include_intermediate=False)

#     c = 1
######## TODO ########
# 1. completly on gpu (For Small Dataset : Yes | Visual Dataset : No)
# 2. weights being sent along with the goal (Not Used at this point)
# 3. reward scale for multistep (Need to Discuss) 
######################
