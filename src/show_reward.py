import numpy as np

def visualize_reward(filename):
    import matplotlib.pyplot as plt

    data = np.loadtxt(filename, delimiter='\t', skiprows=1)

    rewards = data[:, 1]
    stds    = data[:, 2]
    episodes = np.array(list(range(rewards.shape[0]))) + 1
    plt.plot(episodes, rewards)
    plt.fill_between(episodes, rewards-stds, rewards+stds, alpha=0.3)

    plt.xlim((episodes[0], episodes[-1]))

    plt.figure()
    plt.plot(episodes, data[:, 11])
    plt.plot(episodes, data[:, 12])

    plt.show()


if __name__ == "__main__":
    import glob
    filelist = glob.glob('results/progress.txt')
    visualize_reward(filelist[0])

