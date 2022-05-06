from argparse import ArgumentParser
import yaml
import os
from tkinter import *

from simulator.grid_world_testing import GridWorld
from simulator.envs_training import EnvHandler
from algos.vpg import *


parser = ArgumentParser(description = "Testing vacuum cleaning smart agent")

parser.add_argument(
    "--num_cells",
    "-nc",
    type = int,
    default = 5,
    required = False,
    help = "Grid world's side number of cells"
)

parser.add_argument(
    "--cfg",
    type = str,
    default = "/home/tmr/Documents/PhD/My_PhD/WASP-courses/WASP/6_AI_ML/assignment/wasp-AI-ML-m1/cfg/vpg_training.yaml",
    required = False,
    help = "configuration file comprising: networks design choices, hyperparameters, etc."
)

args = parser.parse_args()
device = torch.device("cpu")

assert args.num_cells <= 11 and args.num_cells >= 5, "Please select a grid size between 5 and 11"

print(f"\n========================================Grid world with {args.num_cells} cells========================================\n")

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)
obs_shape = cfg["window_size"] if cfg["window_size"] != args.num_cells else args.num_cells + 2
obs_sp = torch.zeros(obs_shape, obs_shape,3) 
act_sp = torch.zeros(4) # (down, up, right, left)


# Load model 
print("\nLoading model...\n")
model_path = cfg["save_dir"]

eh = EnvHandler(args.num_cells, 
                cfg["obs_reward"],
                cfg["dirt_reward"],
                cfg["ene_reward"],
                cfg["window_size"])

pv = torch.load(os.path.join(cfg["save_dir"], "models", f"pv.pth"))
print("Loaded model!")

# Testing
gw = GridWorld(args.num_cells, 
               pv, 
               eh)

gw.get_world()

gw.root.mainloop()