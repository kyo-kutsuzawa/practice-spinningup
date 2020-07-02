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
    parser.add_argument('--model', type=str, default='results/result/pyt_save/model.pt', help='Model directory')
    parser.add_argument('--norender', action='store_false', dest='render', help='')
    args = parser.parse_args()

    def make_env():
        env = gym.make("Pendulum-v0")
        return env

    # Setup parameters
    model_params = {
        'hidden_sizes': (64, 64),
        'activation': nn.Tanh
    }
    model = torch.load(args.model)

    env = make_env()
    obs = env.reset()

    cnt = 0
    while cnt < 2000:
        cnt += 1
        env.render()

        obs = torch.as_tensor(obs, dtype=torch.float32)
        action = model.act(obs)

        obs, r, done, info = env.step(action)

        if args.render:
            time.sleep(env.dt)


if __name__ == '__main__':
    main()
