import numpy as np
import torch 

def toTensor(array, device):
    if len(array.shape) > 2:
        # pixel_obs
        d_type=torch.uint8
    else:
        d_type=torch.float32

    return torch.tensor(array, dtype=d_type, device=device)
def load_dataset(dataset_path, ob_dtype=np.float32, 
                 action_dtype=np.float32, compact_dataset=False):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
    """
    with np.load(dataset_path, mmap_mode='r') as file:
        dataset = dict()
        for k in ['observations', 'actions', 'terminals']:
            if k == 'observations':
                dtype = ob_dtype
            elif k == 'actions':
                dtype = action_dtype
            else:
                dtype = np.float32
            dataset[k] = file[k][...].astype(dtype, copy=False)
    
    if 'visual' in dataset_path:
        dataset['observations'] = dataset['observations'].transpose(0,3,1,2)
    
    if compact_dataset:

        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
        
        max_size = dataset['observations'].shape[0]
        state_shape = dataset['observations'].shape[1:]
        action_shape = dataset['actions'].shape[1:]
        # Store obs on GPU if they are sufficient small.
        memory, _ = torch.cuda.mem_get_info()
        obs_space = np.prod((max_size, * state_shape), dtype=np.int64) * 1 if ob_dtype == np.uint8 else 4
        ard_space = max_size * (action_shape[0] + 2) * 4
        if obs_space + ard_space < memory:
            storage_device = torch.device('cuda')
        else:
            storage_device = torch.device('cpu')    
        print(f'Storage_device {storage_device}')

        dataset= {k:toTensor(v, storage_device) for k, v in dataset.items()}
    else:

        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        dataset['next_observations'] = dataset['observations'][next_ob_mask]
        dataset['observations'] = dataset['observations'][ob_mask]
        dataset['actions'] = dataset['actions'][ob_mask]
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

        # N X T X H X W X C
        
        terminals  = np.where(dataset['terminals'] == 1)[0]+1
        num_trajectory = len(terminals)
        traj_length = terminals[0]
        for k in dataset:
            dataset[k] = dataset[k].reshape(num_trajectory, traj_length, *dataset[k][0].shape)
        
    return dataset

