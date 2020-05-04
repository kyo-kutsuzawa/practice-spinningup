import gym
from gym import wrappers, logger
import os
import numpy as np


class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


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
    env_name = 'CartPole-v0'

    # Setup an environment and agent
    env = gym.make(env_name)
    agent = RandomAgent(env.action_space)

    if args.mode == 'train':
        logger.set_level(logger.INFO)  # You can set the level to logger.DEBUG or logger.WARN if you want to change the amount of output.
        env = wrappers.Monitor(env, directory=args.out, force=True)

        # Training loop
        for e in range(args.n_episodes):
            # Initialize variables and the environment
            observation = env.reset()

            # Process an episode
            for t in range(args.max_steps):
                action = agent.act(observation)
                observation, reward, done, info = env.step(action)

                if done:
                    print('Episode {:4d} finished after {:3} timesteps'.format(e+1, t+1))
                    break

                if args.render:
                    env.render()
    else:
        # Test loop
        for e in range(args.n_episodes):
            # Initialize variables and the environment
            observation = env.reset()

            # Process an episode
            for t in range(args.max_steps):
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)

                if done:
                    print('Episode finished after {} timesteps'.format(t+1))
                    break

                if args.render:
                    env.render()


if __name__ == '__main__':
    main()

