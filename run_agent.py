from argparse import ArgumentParser
import yaml
from tkinter import *

from simulator.grid_world import GridWorld


parser = ArgumentParser(description = "Vacuum cleaning smart agent")

parser.add_argument(
    "--num_cells",
    "-nc",
    type = int,
    default = 5,
    required = False,
    help = "Grid world's side number of cells"
)


# Testing
# gw = GridWorld(args.num_cells)

# gw.get_world()

# gw.root.mainloop()