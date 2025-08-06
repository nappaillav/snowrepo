import os
with open("key/wandb.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()
os.environ["MUJOCO_GL"]="egl"
import random
import numpy as np
import torch
import ogbench 
import agent as OfflineMRQ
import utils as utils
import evalutils
import ogb_utils
from tqdm import tqdm 
import wandb 
import hydra
from omegaconf import DictConfig, OmegaConf

LARGE_DATASETS = ["visual-humanoidmaze-medium-navigate-v0"]

@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(args: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    # File name and make folders
    if args.project_name == '': args.project_name = f'OfflineMRQ+{args.env}+{args.seed}'
    if not os.path.exists(args.eval_folder): os.makedirs(args.eval_folder)
    if not os.path.exists(args.log_folder): os.makedirs(args.log_folder)
    if args.save_experiment and not os.path.exists(f'{args.save_folder}/{args.project_name}'):
        os.makedirs(f'{args.save_folder}/{args.project_name}')

    utils.set_seed(args.seed)
    if args.debug:
        datapath = args.data_folder + args.env + '-val.npz'
        args.project_name = 'Debug' + args.project_name
    else:
        datapath = args.data_folder + args.env + '.npz'
    if not os.path.exists(datapath):
        ogbench.download_datasets([args.env], dataset_dir=args.data_folder)

    ############# SETUP WANDB TODO ################
    if not args.debug:
        wandb.init(project=args.wandb_project, group='OfflineMRQ', name=args.project_name)
        flat_cfg = OmegaConf.to_container(args, resolve=True)
        # flat_cfg = utils.flatten_dict(flat_cfg)
        wandb.config.update(flat_cfg)
    
    env_name = args.env
    # Change For Frame stacking 
    frame_stack = None if args.frame_stack==0 else args.frame_stack 
    env = ogb_utils.make_env_and_datasets(env_name, frame_stack=frame_stack)

    pixel_obs = True if 'visual' in args.env else False
    if args.frame_stack == 0:
        obs_shape = (3, 64, 64) if 'visual' in args.env else env.observation_space.shape
    else:
        obs_shape = (3*args.frame_stack, 64, 64) if 'visual' in args.env else env.observation_space.shape

    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    print(f"Max Action : {max_action}")
    history = args.frame_stack
    agent = OfflineMRQ.Agent(obs_shape=obs_shape, action_dim=action_dim, max_action=max_action,
                    pixel_obs=pixel_obs, discrete=False, device=device, history=history, hp=dict(args.hp))

    agent.replay_buffer.load_ogbench(dataset_path=datapath)
    print(f"Size of Dataset : {agent.replay_buffer.state.shape} | Length of Dataset : {agent.replay_buffer.num_traj}")

    logger = utils.Logger(f'{args.log_folder}/{args.project_name}.txt')

    max_timesteps = args.total_timesteps
    
    logger.log_print(f"Task {args.env}")
    
    logger.log_print(f"Loaded {datapath}")

    if args.env in LARGE_DATASETS:
        print(f"{args.env} using Chunked Buffer")
        assert args.hp.batched_buffer == True
    else:
        assert args.hp.batched_buffer == False

    evals = []
    for t in range(max_timesteps+1):
    # for t in tqdm(range(max_timesteps+1)):

        evalutils.evaluate_ogbench(agent, env, evals, eval_tasks=None, t=t, 
                                   eval_freq=args.eval_freq, eval_eps=args.eval_eps)
        
        # log_status, metrics = agent.train()
        train_metrics = agent.train()
        if t%args.log_freq == 0: 
            if not args.debug:
                wandb.log(train_metrics, step=t)
            logger_text = utils.log_util(train_metrics, t)
            logger.log_print(logger_text)
        

    # HERE WORK on the LOGGER TODO

if __name__ == "__main__":
    main()

