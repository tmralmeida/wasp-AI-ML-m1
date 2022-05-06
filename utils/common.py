import numpy as np
import matplotlib.pyplot as plt
import os

def stats(arr):
    return np.mean(arr), np.max(arr), np.min(arr)


def plot(ep, stats_return, cfg): 
    mean_ = stats_return["mean"]
    max_ = stats_return["max"]
    min_ = stats_return["min"]
    plt.figure(2)
    plt.clf()
    plt.cla()
    plt.title("Training agent...")
    plt.xlabel(f"# Epochs")
    plt.ylabel("Avg Returns")
    plt.plot(mean_, c = "r")
    plt.fill_between(np.arange(len(mean_)), max_, min_, facecolor = "blue", alpha = 0.3)
    plt.pause(0.001)
    print("Epoch", ep, "\n", "Avg Return", mean_[-1])
    if (ep+1 == cfg["epochs"]) and cfg["save_plot"]:
        print("saving plots...")
        plt.savefig(os.path.join(cfg["save_dir"], "plots", f"plot.png"))
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    
def create_dirs(cfg):
    path = cfg["save_dir"]
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, "models"))
    if cfg["save_plot"] and not os.path.exists(os.path.join(path, "plots")):
        os.makedirs(os.path.join(path, "plots"))
        