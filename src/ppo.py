import gym
from gym import wrappers
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from spinup import ppo_tf1 as ppo
import spinup.algos.tf1.ppo.core as core

eps = 1e-8


def mlp(x, hidden_sizes, out_size, activation, output_activation):
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
    y = tf.layers.dense(x, units=out_size, activation=output_activation)
    return y


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+eps))**2 + 2*log_std * np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]

    mu = mlp(x, hidden_sizes, act_dim, activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std

    logp = gaussian_likelihood(a, mu, log_std)
    log_pi = gaussian_likelihood(pi, mu, log_std)

    return pi, logp, log_pi


def main():
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser(description="Example of Gym + Tensorflow")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Whether training or test')
    parser.add_argument('--n-episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Number of max. steps for an episode')
    parser.add_argument('--render', action='store_true', help='If specified, render the environment')
    parser.add_argument('--out', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    # Define settings
    env_name = 'MountainCarContinuous-v0'

    # Setup parameters
    model_params = {
        'policy': mlp_gaussian_policy,
        'hidden_sizes': (64,),
        'activation': tf.tanh
    }
    ppo_params = {
        'env_fn': lambda: gym.make(env_name),
        #'env_fn': lambda: wrappers.Monitor(gym.make(env_name), directory=args.out, force=True),
        'ac_kwargs': model_params,
        'seed': 0,
        'steps_per_epoch': args.max_steps,
        'epochs': args.n_episodes,
        'logger_kwargs': dict(output_dir=args.out)
    }
    #from spinup.utils.run_utils import setup_logger_kwargs
    #logger_kwargs = setup_logger_kwargs('ppo', ppo_params['seed'])

    # Run PPO
    ppo(**ppo_params)

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

