from argparse import ArgumentParser
from simulator.grid_world import GridWorld
from tkinter import *



parser = ArgumentParser(description = "Vacuum cleaning smart agent")

parser.add_argument(
    "--num_cells",
    "-nc",
    type = int,
    default = 4,
    required = False,
    help = "Grid world's side number of cells"
)


args = parser.parse_args()

assert args.num_cells <= 11 and args.num_cells >= 5, "Please select a grid size between 5 and 11"

print(f"\n========================================Grid world with {args.num_cells} cells========================================\n")

gw = GridWorld(args.num_cells)

gw.get_world()

gw.root.mainloop()