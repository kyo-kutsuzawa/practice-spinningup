import gym
from gym import wrappers
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import time


def main():
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser(description="Tracking Task by a 7 Dof Arm.")
    parser.add_argument('--model', type=str, default='results', help='Model directory')
    parser.add_argument('--norender', action='store_false', dest='render', help='')
    parser.add_argument('--out', type=str, default='angles.csv', help='')
    args = parser.parse_args()

    def make_env():
        env = gym.make("Pendulum-v0")
        #env._max_episode_steps  = 200

        return env

    # Setup parameters
    model_params = {
        'hidden_sizes': (64, 64),
        'activation': nn.Tanh
    }
    model = torch.load(args.model)

    # Setup a record
    record = []

    env = make_env()
    obs = env.reset()
    #record.apuend([env.t] + obs.tolist() + env.fingertip_position.tolist())

    cnt = 0
    while cnt < 2000:
        cnt += 1
        env.render()

        obs = torch.as_tensor(obs, dtype=torch.float32)
        action = model.act(obs)

        #record.append([env.t] + obs.tolist() + action.tolist())

        obs, r, done, info = env.step(action)

        if args.render:
            time.sleep(env.dt)

    # Save the record
    record = np.array(record)
    np.savetxt(args.out, record, delimiter=',')

    #plot(record)


def plot(data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 0    | 1      | ... | n_obs-2 | n_obs-1  | n_obs    | n_obs+1 | ... | n_obs+n_act | n_obs+n_act+1 | n_obs+n_act+2 | n_obs+n_act+3
    # time | joint1 | ... | jointN  | target_x | target_z | torque1 | ... | torqueN     | finger_x      |  finger_y     | finger_z

    n_obs = 3 + 3 + 2
    n_act = 3

    # Plot target positions
    plt.plot(data[:, n_obs-1], data[:, n_obs], color='black')

    # Plot finger positions
    cmap = plt.get_cmap('jet')
    for i in range(data.shape[0]):
        plt.plot(data[i:i+2, n_obs+n_act+1], data[i:i+2, n_obs+n_act+3], color=cmap(i/data.shape[0]))

    # Plot joint angles (3 DoF)
    ax = plt.figure().add_subplot(111, projection='3d')
    for i in range(data.shape[0]):
        ax.plot(data[i:i+2, 1], data[i:i+2, 2], data[i:i+2, 3], color=cmap(i/data.shape[0]))

    # Plot joint torques (actions)
    plt.figure()
    for i in range(n_act):
        plt.plot(data[:, 0], data[:, n_obs+i+1], label="joint {}".format(i))
    plt.xlim((data[0, 0], data[-1, 0]))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
