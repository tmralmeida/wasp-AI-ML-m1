from argparse import ArgumentParser
import yaml
from tkinter import *
import torch.optim as optim
import torch.nn.functional as F

from simulator.grid_world import GridWorld
from simulator.generate_environements import EnvGenerator
from algos.vpg import *
from utils.common import *



parser = ArgumentParser(description = "Vacuum cleaning smart agent")

parser.add_argument(
    "--num_cells",
    "-nc",
    type = int,
    default = 11,
    required = False,
    help = "Grid world's side number of cells"
)


parser.add_argument(
    "--num_envs",
    "-ne",
    type = int,
    default = 50,
    required = False,
    help = "Number of environments to train the agent"
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

obs_sp = torch.zeros(cfg["window_size"],cfg["window_size"],3) # window size, window_size, 3
act_sp = torch.zeros(4) # (down, up, right, left)

# Training agent
pv = PolicyValue(observation_space = obs_sp, action_space = act_sp, hidden_sizes = cfg["hidden_sizes"], device = device)
memory = ReplayMemory(cfg["steps_per_epoch"], 
                      obs_sp.shape, 
                      act_sp.shape, 
                      gamma = cfg["gamma"], 
                      lam = cfg["lambda"])

pi_optimizer = optim.Adam(pv.pi.parameters(), lr = float(cfg["pi_lr"]))
v_optimizer = optim.Adam(pv.v_pi.parameters(), lr = float(cfg["v_lr"]))

ep_ret, stats_return, stats_return["mean"], stats_return["max"], stats_return["min"], all_durations = 0, {}, [], [], [], []

# Generating environment
eg = EnvGenerator(args.num_cells, cfg["window_size"])
eg.create_env()

for epoch in range(cfg["epochs"]):
    epoch_returns = []
    for st in range(cfg["steps_per_epoch"]):
        obs = eg.get_obs()
        action, value = pv.step(torch.as_tensor(obs, dtype = torch.float32).to(device))
        action = F.one_hot(torch.from_numpy(action), num_classes = act_sp.shape[0])
        reward = eg.take_action(action.numpy())
        ep_ret += reward.item()
        memory.push(Experience(obs, action, reward), value)
        
        timeout = st == cfg["max_ep_len"]
        terminal = eg.is_done() or timeout
        epoch_ended = st == cfg["steps_per_epoch"] - 1 
        
        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by episode at %d steps.'%st, flush=True)
                
            # trajectory didn't reach terminal state (not done) --> bootstrap
            if timeout or epoch_ended:
                _, value = pv.step(torch.as_tensor(obs, dtype = torch.float32).to(device))
            else:
                value = 0
            memory.finish_path(value)
            
            if terminal:
                epoch_returns.append(ep_ret)
                all_durations.append(st)
            eg.create_env()
            ep_ret = 0

    # Update VPG
    data = memory.get()
    _obs, _act, _ret, _adv = data["obs"].to(device), data["act"].argmax(dim = -1).to(device), data["ret"].to(device), data["adv"].to(device)
    # Train policy
    pi_optimizer.zero_grad()
    pi, logp = pv.pi(_obs.flatten(1), _act)
    loss_pi = -(logp * _adv).mean()
    loss_pi.backward()
    pi_optimizer.step()
    mean_, max_, min_ = stats(epoch_returns)
    stats_return["mean"].append(mean_)
    stats_return["min"].append(min_)
    stats_return["max"].append(max_)
    
    # Train value function
    for i in range(cfg["v_train_iters"]):
        v_optimizer.zero_grad()
        loss_v_pi = ((pv.v_pi(_obs.flatten(1)) - _ret)**2).mean()
        loss_v_pi.backward()
        v_optimizer.step()

    plot(epoch + 1, stats_return)
    

print("\nTraining finished\n")

# Testing
# gw = GridWorld(args.num_cells)

# gw.get_world()

# gw.root.mainloop()