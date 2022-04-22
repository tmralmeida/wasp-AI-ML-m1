from typing import Tuple, List
import numpy as np
import torch
import random
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt


DEBUG = True

OBSTACLE_REWARD = -1
DIRTY_REWARD = 2
ENGERGY_REWARD = -1

class EnvGenerator():
    def __init__(self, 
                 num_cells, 
                 obs_window_size = 3):
        self.num_cells = num_cells # number of cells
        self.num_obs = self.num_cells
        self.cells_type = ["obs", "robot","dirt"]
        self.window_size = obs_window_size
        
        assert self.window_size % 2 == 1 and self.window_size <= self.num_cells, "Window size not allowed" # it should be odd
        
    
    def __burn_cell(self, loc : Tuple[int, int]):
        self.free_cells.remove(loc)
    
        
    def get_2dloc(self, n : int):
        locs = []
        for _ in range(n):
            l = random.choice(self.free_cells)
            locs.append(l)
            self.__burn_cell(l)
        return locs
    
    
    def update_env(self, env : np.array, graph : List[List], locs : List[Tuple[int]], type : str):
        new_env, new_graph = deepcopy(env), deepcopy(graph)
        for x, y in locs:
            new_env[x, y, self.cells_type.index(type)] = 1 
            new_graph[x][y] = self.cells_type.index(type)
        return new_env, new_graph
    
    
    def get_obs(self):
        x_pt, y_pt = self.robot_loc
        if self.window_size !=  self.num_cells:
            offset = int(self.window_size / 2)
            lu_c, rd_c = ( x_pt - offset, y_pt - offset), (x_pt + offset, y_pt + offset)# left upper corner right down corner
        
        
        obs = self.env[lu_c[0]:rd_c[0]+1,lu_c[1]:rd_c[1]+1,:]
        return torch.from_numpy(obs)
    
    
    
    def is_done(self):
        return self.cnt_dirt == 0 
    
    
    def __getaction_offsets(self, action):
        action = action.argmax()
        if action == 0:
            offset_x, offset_y = -1, 0
        elif action == 1:
            offset_x, offset_y = 1, 0
        elif action == 2:
            offset_x, offset_y = 0, -1
        elif action == 3:
            offset_x, offset_y = 0, 1
        return offset_x, offset_y
    
    
    def take_action(self, action : np.array):
        # action -> 0: up; 1 -> down; 2 -> left; 3 -> right 
        reward = torch.tensor(ENGERGY_REWARD) # the robot moves so he starts already with -1 
        burn_dir, robot_moved = False, False
        
        obs_map = deepcopy(self.env[:,:,0]) # obstacles map
        dirt_map = deepcopy(self.env[:,:, 2]) # dirt map
        ox, oy = self.__getaction_offsets(action)
            
        # compute reward
        if self.robot_loc[0] + ox >= self.num_cells + 2 or self.robot_loc[1] + oy >= self.num_cells + 2: # against a wall
            reward += OBSTACLE_REWARD
            self.cnt_crash += 1
        else:
            if obs_map[self.robot_loc[0] + ox, self.robot_loc[1] + oy] == 1:
                reward += OBSTACLE_REWARD
                self.cnt_crash += 1
            elif dirt_map[self.robot_loc[0] + ox, self.robot_loc[1] + oy] == 1:
                reward += DIRTY_REWARD
                self.cnt_dirt -= 1
                burn_dir = True
                robot_moved = True
            elif obs_map[self.robot_loc[0] + ox, self.robot_loc[1] + oy]  == 0 and dirt_map[self.robot_loc[0] + ox, self.robot_loc[1] + oy] == 0: # free cell
                robot_moved = True
                
            
        # update environment if necessary
        if burn_dir:
            self.env[self.robot_loc[0] + ox, self.robot_loc[1] + oy, 2] = 0
        if robot_moved:
            # update robot loc
            self.env[self.robot_loc[0], self.robot_loc[1], 1] = 0 # robot is not here anymore
            self.robot_loc[0] += ox
            self.robot_loc[1] += oy
            self.env[self.robot_loc[0], self.robot_loc[1], 1] = 1 # place robot
        
        return reward 
    
    
    
    def create_env(self):
        print(f"\n\n=============================================Generating environment=============================================")
        self.cnt_crash, self.free_cells = 0, []
        for i in range(1, self.num_cells + 1):
            for j in range(1, self.num_cells + 1):
                self.free_cells.append((i,j))
        initial_env = np.zeros((self.num_cells + 2, self.num_cells + 2, 3)) # 3 -> robot, obs, dirt
        initial_env[-1, :, self.cells_type.index("obs")] = 1 
        initial_env[0, :, self.cells_type.index("obs")] = 1
        initial_env[:, -1, self.cells_type.index("obs")] = 1
        initial_env[:, 0, self.cells_type.index("obs")] = 1 
        
        initial_graph = (np.ones((self.num_cells + 2, self.num_cells + 2)) * 3)
        initial_graph[-1, :] = 0
        initial_graph[0, :] = 0
        initial_graph[:, -1] = 0
        initial_graph[:, 0] = 0
        initial_graph.tolist()
        # place the robot
        robot_loc = self.get_2dloc(1)
        self.robot_loc = list(robot_loc[-1])
        print("\nRobot location:", robot_loc)
        new_env_robot_loc, new_graph_robot_loc = self.update_env(initial_env, initial_graph, robot_loc, "robot")

        # place obstacles 
        obs_loc = self.get_2dloc(self.num_obs) 
        print("\nObstacles locations:", obs_loc)
        print("\nAfter placing obstacles, free cells list:", self.free_cells)
        new_env_obs_loc, new_graph_obs_loc = self.update_env(new_env_robot_loc, new_graph_robot_loc, obs_loc, "obs")
        print("\nGraph after placing obstacles:\n", new_graph_obs_loc)
        
        
        # compute pssible paths from the robot's location
        set_covered_cells = set()
        for (end_point_x, end_point_y) in self.free_cells:
            curr_graph = deepcopy(new_graph_obs_loc)
            curr_graph[end_point_x][end_point_y] = 2
            if findPath(curr_graph)[0]:
                set_covered_cells.add(findPath(curr_graph)[1])
        print("\nCovered cells", set_covered_cells)
        
        if len(set_covered_cells) > 0:    
            # place dirty
            n_dirt = random.randint(self.num_obs, len(set_covered_cells))
            self.cnt_dirt = n_dirt
            free_cells = list(deepcopy(set_covered_cells))
            dirt_locs = []
            for _ in range(n_dirt):
                l = random.choice(free_cells)
                dirt_locs.append(l)
                free_cells.remove(l)
                #self.__burn_cell(l)

            print("\nWe are going to place dirt at", dirt_locs)
            self.env, new_graph_dirt_loc = self.update_env(new_env_obs_loc, new_graph_obs_loc, dirt_locs, "dirt")
            print("\nGraph after placing dirt\n", new_graph_dirt_loc)
            
            # debug
            if DEBUG: 
                image_r = np.zeros((self.num_cells + 2, self.num_cells + 2))
                image_g = np.zeros((self.num_cells + 2, self.num_cells + 2))
                image_b = np.zeros((self.num_cells + 2, self.num_cells + 2))
                image_r[new_env_obs_loc[:,:,0] == 1] = 255
                image_g[new_env_obs_loc[:,:,2] == 1] = 255
                image_b[new_env_obs_loc[:,:,1] == 1] = 255
                image = np.stack([image_r, image_g, image_b], axis = 2).astype(int)
                fig = plt.figure(figsize=(10, 10))
                fig.add_subplot(1, 3, 1)
                plt.imshow(image);
                plt.title("Initial Environment");
                
                image_covered = deepcopy(image)
                for covered_cell in set_covered_cells:
                    xx, yy = covered_cell[0], covered_cell[1] 
                    image_covered[xx, yy, 1] = 255
                
                fig.add_subplot(1, 3, 2)
                plt.imshow(image_covered);
                plt.title("Cells Covered");
                
                
                image_dirt = deepcopy(image_covered)
                for dl in dirt_locs:
                    xx_d, yy_d = dl[0], dl[1]
                    image_dirt[xx_d, yy_d, 0] = 255 
                fig.add_subplot(1, 3, 3)
                plt.imshow(image_dirt);
                plt.title("Final Environment");
                plt.show();
        else:
            self.create_env()
            
            
            
        
        

# code based on: https://www.geeksforgeeks.org/find-whether-path-two-cells-matrix/
class Graph:
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.free_cells = []
        for i in range(n):
            for j in range(n):
                self.free_cells.append((i,j))
     
    # add edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
 
    # BFS function to find path from source to sink    
    def BFS(self, s, d):
         
        # Base case
        if s == d: # the dirt can't be in the robot's currrent location
            return False, None
             
        # Mark all the vertices as not visited
        visited = [False]*(self.n*self.n + 1)
 
        # Create a queue for BFS
        queue = []
        queue.append(s)
 
        # Mark the current node as visited and
        # enqueue it
        visited[s] = True
        while(queue):
 
            # Dequeue a vertex from queue
            s = queue.pop(0)
 
            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent has
            # not been visited, then mark it visited
            # and enqueue it
            for i in self.graph[s]:
                 
                # If this adjacent node is the destination
                # node, then return true
                if i == d:
                    return True, self.free_cells[d - 1]
 
                # Else, continue to do BFS
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
 
        # If BFS is complete without visiting d
        return False, None
 
def isSafe(i, j, matrix):
    if i >= 0 and i <= len(matrix) and j >= 0 and j <= len(matrix[0]):
        return True
    else:
        return False
    

def findPath(M):
    s, d = None, None # source and destination
    N = len(M)
    g = Graph(N)
 
    # create graph with n * n node
    # each cell consider as node
    k = 1 # Number of current vertex
    for i in range(N):
        for j in range(N):
            if (M[i][j] != 0):
 
                # connect all 4 adjacent cell to
                # current cell
                if (isSafe(i, j + 1, M)):
                    g.addEdge(k, k + 1)
                if (isSafe(i, j - 1, M)):
                    g.addEdge(k, k - 1)
                if (isSafe(i + 1, j, M)):
                    g.addEdge(k, k + N)
                if (isSafe(i - 1, j, M)):
                    g.addEdge(k, k - N)
             
            if (M[i][j] == 1):
                s = k
 
            # destination index    
            if (M[i][j] == 2):
                d = k
            k += 1
 
    # find path Using BFS
    return g.BFS(s, d)