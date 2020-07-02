import numpy as np
import matplotlib.pyplot as plt


def visualize_reward(results, figname):
    for filename, label in results:
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)

        rewards = data[:, 1]
        stds    = data[:, 2]
        episodes = np.array(list(range(rewards.shape[0]))) + 1

        plt.plot(episodes, rewards, label=label)
        plt.fill_between(episodes, rewards-stds, rewards+stds, alpha=0.3)

        plt.xlim((1, 3000))
        plt.ylim((-1800, 0))

    plt.legend(fontsize=18)

    plt.savefig(figname)
    plt.show()


if __name__ == "__main__":
    # default
    results = [
        ("results/result1/progress.txt", "default"),
    ]
    visualize_reward(results, "default.svg")

    # NN size
    results = [
        ("results/result1/progress.txt", "default (2lys)"),
        ("results/result6/progress.txt", "larger (4lys)"),
        ("results/result5/progress.txt", "smaller (1lys)"),
    ]
    visualize_reward(results, "nnsize.svg")

    # discount rate
    results = [
        ("results/result1/progress.txt",  "default (0.99)"),
        ("results/result14/progress.txt", "gamma=0.999"),
        ("results/result2/progress.txt",  "gamma=0.9"),
        ("results/result15/progress.txt", "gamma=0.7"),
    ]
    visualize_reward(results, "gamma.svg")

    # lambda
    results = [
        ("results/result1/progress.txt",  "default (0.97)"),
        ("results/result13/progress.txt", "lam=0.999"),
        ("results/result8/progress.txt",  "lam=0.7"),
    ]
    visualize_reward(results, "lam.svg")

    # learning rate
    results = [
        ("results/result1/progress.txt",  "default"),             # 0.0003, 0.001
        ("results/result12/progress.txt", "10x vf_lr"),           # 0.0003, 0.01
        ("results/result10/progress.txt", "10x pi_lr"),           # 0.003,  0.001
        ("results/result9/progress.txt",  "10x pi_lr & vf_lr"),   # 0.003,  0.01
        ("results/result11/progress.txt", "0.1x vf_lr"),          # 0.0003, 0.0001
    ]
    visualize_reward(results, "lr.svg")
