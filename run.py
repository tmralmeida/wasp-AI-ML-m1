from argparse import ArgumentParser
from simulator.grid_world import GridWorld
from simulator.generate_environements import EnvGenerator
from tkinter import *



parser = ArgumentParser(description = "Vacuum cleaning smart agent")

parser.add_argument(
    "--num_cells",
    "-nc",
    type = int,
    default = 5,
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



args = parser.parse_args()

assert args.num_cells <= 11 and args.num_cells >= 5, "Please select a grid size between 5 and 11"

print(f"\n========================================Grid world with {args.num_cells} cells========================================\n")



# Generating environments:
for _ in range(args.num_envs):
    eg = EnvGenerator(args.num_cells, 
                    args.num_envs)
    eg.get_env()
    



# Training agent





# gw = GridWorld(args.num_cells)

# gw.get_world()

# gw.root.mainloop()