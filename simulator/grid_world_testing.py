from copy import deepcopy
import numpy as np
from tkinter import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import time



ROBOT_VALUE = 0
OBSTACLE_VALUE = -10
DIRTY_VALUE = 1
FREE_VALUE = -1


class GridWorld():
    def __init__(self, 
                 nc : int, 
                 pv : nn.Module, 
                 eh : object) -> None:
        self.side = nc*100
        self.nc = nc # number of cells
        self.no = self.nc # number max of obstacles
        self.nr = 1 # number maximum of robots
        self.root = Tk()    
        self.root.minsize(self.side, self.side)     
        self.root.title("Grid World for vacuum cleaning")  
        self.root.resizable(0,0) # no maximize
        self.__init_vars()
        self.pv  = pv # trained model
        self.event_handler = eh
        self.cells_type = ["obs", "robot","dirt"]
        self.device = torch.device("cpu")
        
    
    def __init_vars(self):
        self.clicks_robot, self.clicks_obstacles, self.clicks_dirty = 0, 0, 0
        self.model = np.ones((self.nc, self.nc)) * FREE_VALUE 
        #print("\nInitial state:\n", self.model)


    def __update_cell(self, x : int, y : int, type : str):
        if type == "robot":
            self.model[y, x] = ROBOT_VALUE if self.model[y, x] == FREE_VALUE else FREE_VALUE 
        elif type == "obs": # obstacle
            self.model[y, x] = OBSTACLE_VALUE if self.model[y, x] == FREE_VALUE else FREE_VALUE
        elif type == "dirty":
            self.model[y, x] = DIRTY_VALUE if self.model[y, x] == FREE_VALUE  else FREE_VALUE 

            
    def __occ_robot(self, x : int, y : int):
        return self.model[y, x] == ROBOT_VALUE
    

    def __occ_obstacle(self, x : int, y : int):
        return self.model[y, x] == OBSTACLE_VALUE

    
    def __occ_dirty(self, x : int, y : int):
        return self.model[y, x] == DIRTY_VALUE


    def __free_cell(self, x : int, y : int):
        return not self.model[y, x] == ROBOT_VALUE and not self.model[y, x] == OBSTACLE_VALUE and not self.model[y, x] == DIRTY_VALUE


    def coord_robot(self, event):
        pix_x = event.x         
        pix_y = event.y
        coord_x = int(pix_x / 100)
        coord_y = int(pix_y / 100)
        x0_paint,y0_paint = coord_x * 100 + 3, coord_y * 100 + 3
        x1_paint,y1_paint = x0_paint + 94, y0_paint + 94
        if self.clicks_robot < self.nr and self.__free_cell(coord_x, coord_y):
            self.clicks_robot += 1        
            self.canvas.create_rectangle(x0_paint,y0_paint,x1_paint,y1_paint, outline="#000", fill="#000") # painting black
            # updating cell in the state 
            self.__update_cell(coord_x, coord_y, "robot")
            #print("Placing robot at", (coord_y, coord_x))
            #print("New model\n", self.model)
            
        else:
            # if it is occupied by the robot -> remove the robot
            if self.__occ_robot(coord_x, coord_y):
                self.clicks_robot -= 1
                self.canvas.create_rectangle(x0_paint,y0_paint,x1_paint,y1_paint, outline="#fff", fill="#fff") # re-painting white
                #print("Removing robot from", (coord_y, coord_x))
                self.__update_cell(coord_x, coord_y, "robot")
                #print("New model\n", self.model)
            else:
                print("Not occupied by the robot!")
           
           
               
    def coords_obs_dirt(self, event):
        pix_x = event.x         # Get x and y co-ordinate of the click event
        pix_y = event.y
        coord_x = int(pix_x / 100)
        coord_y = int(pix_y / 100)
        x0_paint,y0_paint = coord_x * 100 + 3, coord_y * 100 + 3
        x1_paint,y1_paint = x0_paint + 94, y0_paint + 94
        if self.clicks_obstacles < self.no and self.__free_cell(coord_x, coord_y):
            self.clicks_obstacles += 1
            self.canvas.create_rectangle(x0_paint,y0_paint,x1_paint,y1_paint, outline="#f50", fill="#f50") # painting black
            # updating cell in the state 
            self.__update_cell(coord_x, coord_y, "obs")
            #print(f"Placing obstacles at", (coord_y, coord_x))
            #print("New model\n", self.model)
        
        else:
            if self.__occ_obstacle(coord_x, coord_y): # occupied by an obstacle
                self.clicks_obstacles -= 1
                self.canvas.create_rectangle(x0_paint,y0_paint,x1_paint,y1_paint, outline="#fff", fill="#fff") # re-painting white
                #print("Removing obstacle from", (coord_y, coord_x))
                self.__update_cell(coord_x, coord_y, "obs")
                #print("New model\n", self.model)
            
            elif self.__occ_dirty(coord_x, coord_y): # occupied by dirty
                self.clicks_dirty -= 1
                self.canvas.create_rectangle(x0_paint,y0_paint,x1_paint,y1_paint, outline="#fff", fill="#fff") # re-painting white
                #print("Removing dirty from", (coord_y, coord_x))
                self.__update_cell(coord_x, coord_y, "dirty")
                #print("New model\n", self.model)
            
            elif self.__occ_robot(coord_x, coord_y): # occupied by the robot
                print("Not occupied by obstacles or dirty. To remove the robot please use left button of your mouse!")
            
            elif  self.__free_cell(coord_x, coord_y): # let's start placing dirty
                self.clicks_dirty += 1
                self.canvas.create_rectangle(x0_paint,y0_paint,x1_paint,y1_paint, outline="#00ff7f", fill="#00ff7f") 
                self.__update_cell(coord_x, coord_y, "dirty")
                #print(f"Placing dirty at", (coord_y, coord_x))
                #print("New model\n", self.model)
            
            else:
                raise Exception("Designer needs to double check button right")
                
        

    def play_sim(self, event):
        if self.clicks_obstacles == self.no and self.clicks_robot == self.nr and self.clicks_dirty > 0:
            # unbinding buttons
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.canvas.unbind("<p>")
            
            
            print(f"Testing with {self.clicks_dirty} dirty cells and {self.clicks_obstacles} obstacles...")
            
            model_dum = np.ones((self.nc + 2, self.nc + 2)) * (OBSTACLE_VALUE)
            model_dum[1:-1, 1:-1] = deepcopy(self.model)
            
            
            
            # parsing environment to eh format
            initial_env = np.zeros((self.nc + 2, self.nc + 2, 3)) # 3 -> robot, obs, dirt
            # 1)walls
            initial_env[-1, :, self.cells_type.index("obs")] = 1 
            initial_env[0, :, self.cells_type.index("obs")] = 1
            initial_env[:, -1, self.cells_type.index("obs")] = 1
            initial_env[:, 0, self.cells_type.index("obs")] = 1 
            
            # 2)robot
            initial_env[model_dum == ROBOT_VALUE, self.cells_type.index("robot")] = 1
            
            # 3)obstacles
            initial_env[model_dum == OBSTACLE_VALUE, self.cells_type.index("obs")] = 1
            
            # 4)dirt
            initial_env[model_dum == DIRTY_VALUE, self.cells_type.index("dirt")] = 1
            
            robot_loc = np.argwhere(initial_env[:, :, self.cells_type.index("robot")] == 1)[-1]
            
            
            # while loop -> end of the episode 
            ep_ret, ep_len, act_sp = 0, 0, torch.zeros(4)
            self.event_handler.env = initial_env
            self.event_handler.init_vars()
            self.event_handler.robot_loc, self.event_handler.cnt_dirt = robot_loc, self.clicks_dirty
            while (not self.event_handler.is_done()) and (ep_len < self.nc * self.nc * 2):
                prev_robot_loc = deepcopy(self.event_handler.robot_loc)
                obs = self.event_handler.get_obs()
                action, _ = self.pv.step(torch.as_tensor(obs, dtype = torch.float32).to(self.device))
                action = F.one_hot(torch.from_numpy(action), num_classes = act_sp.shape[0])
                reward = self.event_handler.take_action(action.numpy())
                ep_ret += reward.item()
                ep_len += 1
                
                x0_paint_prev,y0_paint_prev = (prev_robot_loc[0] - 1) * 100 + 3, (prev_robot_loc[1] - 1) * 100 + 3
                x1_paint_prev,y1_paint_prev = x0_paint_prev + 94, y0_paint_prev + 94
                x0_paint,y0_paint = (self.event_handler.robot_loc[0] - 1) * 100 + 3, (self.event_handler.robot_loc[1] - 1) * 100 + 3
                x1_paint,y1_paint = x0_paint + 94, y0_paint + 94
                
                if (self.event_handler.robot_loc == prev_robot_loc).all(): # hit in an obstacle do nothing
                    print("\nThe agent hit an obstacle :(")
                else: 
                    self.canvas.create_rectangle(y0_paint_prev,x0_paint_prev,y1_paint_prev, x1_paint_prev, outline="#fff", fill="#fff") # re-painting white
                    self.canvas.create_rectangle(y0_paint, x0_paint,y1_paint, x1_paint, outline="#000", fill="#000") # painting black
                    self.root.update()

                
                time.sleep(0.5)
            if ep_len >= self.nc * self.nc * 2:
                print("\nSTOPPED BY TIME OUT!!! It's a rather complex scenario...\n")
            
            
            
            print("\n==============================STATS==================================\n")
            print(f"\tEpisode return: {ep_ret} in {ep_len} steps!")
            print(f"\t# crashes: {self.event_handler.cnt_crash} -------------- # cleaned dirty cells {self.event_handler.cleaned_cells}")
            
            self.root.destroy()
            
        else:
            print("Please, finish the environment designing!")
    


    def get_world(self):
        self.canvas = Canvas(self.root,height = self.side,width = self.side,bd=0,bg="#fff")
        self.canvas.pack(anchor=CENTER)
        
        # drawing walls
        # canvas.create_line(0,0,self.side, 0, self.side, self.side, 0, self.side, width=5)

        
        # drawing lines
        for x in range(self.nc + 1):
            # vertical line
            self.canvas.create_line(100*x,0,100*x,self.side,width=5)
            for y in range(self.nc + 1):
                # horizontal line
                self.canvas.create_line(0,100*y,self.side, 100*y,width=5)
        self.canvas.focus_set() 
        self.canvas.bind("<Button-1>", self.coord_robot)
        self.canvas.bind("<Button-3>", self.coords_obs_dirt)
        self.canvas.bind("<p>", self.play_sim)
        print("\nPlease place the robot with the left button of your mouse.")

        

