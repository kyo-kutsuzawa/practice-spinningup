import gym
from gym import wrappers
import numpy as np


def main():
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser(description="Example of Gym + Tensorflow")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Whether training or test')
    parser.add_argument('--n-episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Number of max. steps for an episode')
    args = parser.parse_args()

    # Define settings
    env_name = 'CartPole-v0'

    # Setup an environment and agent
    env = gym.make(env_name)

    if args.mode == 'train':
        # Training loop
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
            
                env.render()
    else:
        pass


if __name__ == '__main__':
    main()

