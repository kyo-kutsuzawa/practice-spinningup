import gym
from gym import wrappers
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from spinup import vpg_pytorch as vpg
import spinup.algos.pytorch.vpg.core as core


def main():
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser(description="Example of Gym + PyTorch")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Whether training or test')
    parser.add_argument('--n-episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=200, help='Number of max. steps for an episode')
    parser.add_argument('--render', action='store_true', help='If specified, render the environment')
    parser.add_argument('--out', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    # Define settings
    env_name = 'Pendulum-v0'

    # Setup parameters
    model_params = {
        'hidden_sizes': (64, 64),
        'activation': nn.Tanh
    }
    vpg_params = {
        'env_fn': lambda: wrappers.Monitor(gym.make(env_name), directory=args.out, force=True),
        'ac_kwargs': model_params,
        'seed': 0,
        'steps_per_epoch': 2000,
        'epochs': args.n_episodes,
        'gamma': 0.9,
        'pi_lr': 0.0003,
        'vf_lr': 0.001,
        'max_ep_len': args.max_steps,
        'logger_kwargs': dict(output_dir=args.out)
    }
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs('vpg', vpg_params['seed'])

    # Run VPG
    vpg(**vpg_params)

    # Get scores from last five epochs to evaluate success.
    data = pd.read_table(os.path.join(args.out,'progress.txt'))

    # Make a graph of reward progress
    plt.plot(np.array(list(range(args.n_episodes)))+1, data['AverageEpRet'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xlim((1, args.n_episodes))
    plt.show()


if __name__ == '__main__':
    main()

