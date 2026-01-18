#here we will create the grid (environments with different complexities)

'''
Design coiches:
Grid is a 2d numpy array grid[row, col]

Cells have different states:
0 = free cell
1 = obstacle
2 = start
3 = target

State of the agent is defined by:
(row, col, vs, vy)

acceleration:
ax, ay is either +-1 or 0 

speed is limited to 3 in each direction

vx and vy cannot both be 0 at the same time
'''


#imports
import numpy as np
from typing import Tuple



#action (move in x and y direction by ax, ay)
action = Tuple[int, int]

#state defined by (row, col, vx, vy)
state = Tuple[int, int, int, int]


#define environment class
class env:
    
    free = 0
    obstacle = 1
    start = 2
    target = 3
    
    def __init__(self, grid: np.ndarray, seed: int):
        #store grid
        if grid.ndim != 2:
            raise ValueError("Grid must be a 2D numpy array.")
        self.grid = grid
        #store dimensions
        self.n_rows = grid.shape[0]
        self.n_cols = grid.shape[1]
        #create rng
        self.rng = np.random.default_rng(seed)
        #initialise starting state
        self.state = None
        
        #get start cells
        self.start_cells = np.argwhere(self.grid == self.start)
        #get target cells
        self.target_cells = np.argwhere(self.grid == self.target)
        
        #define possible actions
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]

    #start a new episode (from a random start cell)
    def reset(self):
        #select a random start cell
        start_row, start_col = self.choose_rand_start_cell()
        #initialise state
        self.state = (start_row, start_col, 0, 0)
        
        
        
        
    #helper functions
    def choose_rand_start_cell(self):
        idx = self.rng.integers(0, len(self.start_cells)) #select one random index from 0 to len(start_cells)-1
        start_row, start_col = self.start_cells[idx] #get the corresponding start cell
        return start_row, start_col