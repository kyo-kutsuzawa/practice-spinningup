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
    parser.add_argument('--model',           choices=['model1', 'model2', 'model3'], default='model1', help='Whether model to use')
    parser.add_argument('--render',          action='store_true',         help='If specified, render the environment')
    parser.add_argument('--out',             type=str,   default='results/result', help='Output directory')
    parser.add_argument('--steps-per-epoch', type=int,   default=4000,   help='Number of steps in an epoch')
    parser.add_argument('--epochs',          type=int,   default=1000,   help='Number of training epochs')
    parser.add_argument('--gamma',           type=float, default=0.99,   help='')
    parser.add_argument('--pi-lr',           type=float, default=0.0003, help='')
    parser.add_argument('--vf-lr',           type=float, default=0.001,  help='')
    parser.add_argument('--train-v-iters',   type=int,   default=80,     help='')
    parser.add_argument('--lam',             type=float, default=0.97,   help='')
    parser.add_argument('--max-ep-len',      type=int,   default=1000,   help='Number of max. steps for an episode')
    args = parser.parse_args()

    # Environment-making function
    def make_env():
        env_name = 'Pendulum-v0'
        env = gym.make(env_name)
        wrappers.Monitor(env, directory=args.out, force=True)
        return env

    # Setup parameters
    if args.model == 'model1':
        model_params = {
            'hidden_sizes': (64, 64),
            'activation': nn.Tanh
        }
    elif args.model == 'model2':
        model_params = {
            'hidden_sizes': (32,),
            'activation': nn.Tanh
        }
    elif args.model == 'model3':
        model_params = {
            'hidden_sizes': (128, 128, 128, 128),
            'activation': nn.Tanh
        }

    # Setup VPG parameters
    vpg_params = {
        'env_fn': make_env,
        'ac_kwargs': model_params,
        'seed': 0,  # Manually set a seed value for reproducibility
        'steps_per_epoch': args.steps_per_epoch,
        'epochs': args.epochs,
        'gamma': args.gamma,
        'pi_lr': args.pi_lr,
        'vf_lr': args.vf_lr,
        'train_v_iters': args.train_v_iters,
        'lam': args.lam,
        'max_ep_len': args.max_ep_len,
        'logger_kwargs': dict(output_dir=args.out)
    }
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs('vpg', vpg_params['seed'])

    # Run VPG
    vpg(**vpg_params)

    # Make a reward-progress graph
    data = pd.read_table(os.path.join(args.out,'progress.txt'))
    plt.plot(np.array(list(range(args.epochs)))+1, data['AverageEpRet'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xlim((1, args.epochs))
    plt.show()


if __name__ == '__main__':
    main()
