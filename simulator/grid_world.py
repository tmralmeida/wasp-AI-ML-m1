import numpy as np
import tkinter
from tkinter import *


class GridWorld():
    def __init__(self, 
                 nc : int) -> None:
        self.side = nc*100
        self.nc = nc # number of cells
        self.no = int(self.nc/2)
        self.root = Tk()    
        self.root.minsize(self.side, self.side)     
        self.root.title("Grid World for vacuum cleaning")  
        self.root.resizable(0,0) # no maximize
        self.__init_vars()
        
    def __init_vars(self):
        self.clicks_robot, self.clicks_obstacles, self.clicks_dirty = 0, 0, 0
        self.state = np.zeros(self.nc)
        

    def coord_robot(self, event):
        pix_x = event.x         # Get x and y co-ordinate of the click event
        pix_y = event.y
        coord_x = int(pix_x / 100)
        coord_y = int(pix_y / 100)
        print(f"Updating robot to x = {coord_x} and y = {coord_y}. Thank you for placing the robot!\n \
                Now, please place {self.no} obstacles with the right button of your mouse.")

    
    def coords_obs_dirt(self, event):
        pix_x = event.x         # Get x and y co-ordinate of the click event
        pix_y = event.y
        coord_x = int(pix_x / 100)
        coord_y = int(pix_y / 100)
        print(f"Updating obstacles or dirty to x = {coord_x} and y = {coord_y}")
        

    def get_world(self):
        canvas = Canvas(self.root,height = self.side,width = self.side,bd=0)
        canvas.pack(anchor=CENTER)
        
        # drawing walls
        # canvas.create_line(0,0,self.side, 0, self.side, self.side, 0, self.side, width=5)

        
        # drawing lines
        for x in range(self.nc + 1):
            # vertical line
            canvas.create_line(100*x,0,100*x,self.side,width=5)
            for y in range(self.nc + 1):
                # horizontal line
                canvas.create_line(0,100*y,self.side, 100*y,width=5)
            
        canvas.bind("<Button-1>", self.coord_robot)
        canvas.bind("<Button-3>", self.coords_obs_dirt)
        print("\nPlease place the robot with the left button of your mouse.")

        

