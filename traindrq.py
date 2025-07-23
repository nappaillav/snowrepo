# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# import argparse
# import dataclasses
# from dataclasses import asdict
import os
with open("key/wandb.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()
os.environ["MUJOCO_GL"]="egl"
# import pickle
# import time
import random
import numpy as np
import torch
import ogbench 
from drqv2 import DrQV2Agent
import common.utils as utils
import common.evalutils as evalutils
from tqdm import tqdm 
import wandb 
import hydra
from omegaconf import DictConfig, OmegaConf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(env_name, **env_kwargs):
    return ogbench.make_env_and_datasets(env_name, **env_kwargs)

# @dataclasses.dataclass
# class DefaultExperimentArguments:
#     Atari_total_timesteps: int = 25e5
#     Atari_eval_freq: int = 1e5

#     Dmc_total_timesteps: int = 5e5
#     Dmc_eval_freq: int = 5e3

#     Gym_total_timesteps: int = 1e6
#     Gym_eval_freq: int = 5e3

#     def __post_init__(self): utils.enforce_dataclass_type(self)



@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(args: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    # default_arguments = DefaultExperimentArguments()
    # env_type = args.env.split('-',1)[0]
    # if args.total_timesteps == -1: args.total_timesteps = default_arguments.__dict__[f'{env_type}_total_timesteps']
    # if args.eval_freq == -1: args.eval_freq = default_arguments.__dict__[f'{env_type}_eval_freq']

    # File name and make folders
    if args.project_name == '': args.project_name = f'DRQ-v2+{args.env}+{args.seed}'
    if not os.path.exists(args.eval_folder): os.makedirs(args.eval_folder)
    if not os.path.exists(args.log_folder): os.makedirs(args.log_folder)
    if args.save_experiment and not os.path.exists(f'{args.save_folder}/{args.project_name}'):
        os.makedirs(f'{args.save_folder}/{args.project_name}')

    set_seed(args.seed)
    if args.debug:
        datapath = args.data_folder + args.env + '-val.npz'
        args.project_name = 'Debug' + args.project_name
    else:
        datapath = args.data_folder + args.env + '.npz'
    if not os.path.exists(datapath):
        ogbench.download_datasets([args.env], dataset_dir=args.data_folder)

    ############# SETUP WANDB TODO ################
    if not args.debug:
        wandb.init(project=args.wandb_project, group='MRQ', name=args.project_name)
        flat_cfg = vars(args)
        wandb.config.update(flat_cfg)
    
    env_name = args.env
    env = ogbench.make_env_and_datasets(dataset_name=env_name, env_only=True)
    # env.action_space.seed(args.seed)
    pixel_obs = True if 'visual' in args.env else False
    obs_shape = (3, 64, 64) if 'visual' in args.env else env.observation_space.shape
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    
    agent = DrQV2Agent(obs_shape, action_dim, device, 3e-4, 512,
                        512, 0.01, 2000,
                        2, 0.3, 0.3, True)

    agent.replay_buffer.load_ogbench(dataset_path=datapath)
    print(f"Size of Dataset : {agent.replay_buffer.state.shape} | Length of Dataset : {agent.replay_buffer.num_traj}")

    logger = utils.Logger(f'{args.log_folder}/{args.project_name}.txt')

    max_timesteps = args.total_timesteps
    
    logger.log_print(f"Task {args.env}")
    
    
    logger.log_print(f"Loaded {datapath}")
    
    evals = []
    for t in tqdm(range(max_timesteps+1)):
        
        evalutils.evaluate_ogbench(agent, env, evals, eval_tasks=None, t=t, 
                                   eval_freq=args.eval_freq, eval_eps=args.eval_eps)
        
        # log_status, metrics = agent.train()
        train_metrics = agent.update()
        if t%args.log_freq == 0: 
            if not args.debug:
                wandb.log(train_metrics, step=t)
            # logger_text = utils.log_util(train_metrics, t)
            logger.log_print(str(train_metrics))

if __name__ == "__main__":
    main()

