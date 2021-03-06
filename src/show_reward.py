import numpy as np

def visualize_reward(filelist):
    import matplotlib.pyplot as plt

    for filename in filelist:
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)

        rewards = data[:, 1]
        stds    = data[:, 2]
        episodes = np.array(list(range(rewards.shape[0]))) + 1
        plt.plot(episodes, rewards, label=filename)
        plt.fill_between(episodes, rewards-stds, rewards+stds, alpha=0.3)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    import glob
    filelist = glob.glob('results/*/progress.txt')
    visualize_reward(filelist)
