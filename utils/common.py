import numpy as np
import matplotlib.pyplot as plt

def stats(arr):
    return np.mean(arr), np.max(arr), np.min(arr)


def plot(ep, stats_return):
    mean_ = stats_return["mean"]
    max_ = stats_return["max"]
    min_ = stats_return["min"]
    plt.figure(2)
    plt.clf()
    plt.cla()
    plt.title("Training...")
    plt.xlabel(f"# Epochs")
    plt.ylabel("Avg Returns")
    plt.plot(mean_, c = "r")
    plt.fill_between(np.arange(len(mean_)), max_, min_, facecolor = "blue", alpha = 0.3)
    plt.pause(0.001)
    print("Environment", ep, "\n", "Avg Return", mean_[-1])
    plt.show()