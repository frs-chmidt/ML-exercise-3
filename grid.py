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
from typing import Tuple, List, Any, Dict



#action (move in x and y direction by ax, ay)
action = Tuple[int, int]

#state defined by (row, col, vx, vy)
state = Tuple[int, int, int, int]


#define environment class=============================================================
class env:
    
    free = 0
    obstacle = 1
    start = 2
    target = 3
    
    #constructor_______________________________________________________________________
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
        #validate existence
        if len(self.start_cells) == 0:
            raise ValueError("Grid has no start cells (value 2).")
        #get target cells
        self.target_cells = np.argwhere(self.grid == self.target)
        #validate existence
        if len(self.target_cells) != 1:
            raise ValueError("Grid must have exactly one target cell (value 3).")
        
        #define possible actions
        self.action_space = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]


    #start a new episode (from a random start cell)______________________________________
    def reset(self)-> state:
        #select a random start cell
        start_row, start_col = self.choose_rand_start_cell()
        #initialise state
        self.state = (start_row, start_col, 0, 0)
        return self.state
    
    
    #step function_______________________________________________________________________
    def step(self, action: action):
        
        #if state is None we need to reset the environment first (this selects a random start cell)
        if self.state is None:
            raise ValueError("Environment not reset. Call env.reset() before stepping.")
        
        ax, ay = action
        #check for valid action
        if ax not in (-1, 0, 1) or ay not in (-1, 0, 1):
            raise ValueError("Invalid action. ax and ay must be -1, 0, or 1.")
        
        #get current state
        row0, col0, vx0, vy0 = self.state
        
        #update velocity using action but limit vx and vy to +-3
        vx1 = int(np.clip(vx0 + ax, -2, 2))
        vy1 = int(np.clip(vy0 + ay, -2, 2))
        
        #check that velocity is not both 0        
        if vx1 == 0 and vy1 == 0 and not self.is_start_cell(row0, col0):
            #if both are 0 and we are not in a start cell, keep previous velocity
            vx1, vy1 = vx0, vy0
        
        #update position
        row1 = row0 + vy1
        col1 = col0 + vx1
        
        info = self.trace((row0, col0), (row1, col1))
        
        reward = -1
        
        if info["hit_target"] == True:
            self.state = (info["end_position"][0], info["end_position"][1], vx1, vy1)
            done = True
            return self.state, reward, done, info
        
        if info["collision"] == True:
            #reset to starting cell
            start_row, start_col = self.choose_rand_start_cell()
            self.state = (start_row, start_col, 0, 0)
            done = False
            return self.state, reward, done, info
        
        self.state = (row1, col1, vx1, vy1)
        done = False
        return self.state, reward, done, info
    
    #simple helper functions_______________________________________________________________________
    
    #choose a random start cell from the list of start cells
    def choose_rand_start_cell(self):
        idx = self.rng.integers(0, len(self.start_cells)) #select one random index from 0 to len(start_cells)-1
        start_row, start_col = self.start_cells[idx] #get the corresponding start cell
        return int(start_row), int(start_col)
    
    #return true if cell is a start cell (its value is 2)
    def is_start_cell(self, row: int, col: int) -> bool:
        return self.grid[row, col] == self.start
    
    #return true if cell is a target cell (its value is 3)
    def is_target_cell(self, row: int, col: int) -> bool:
        return self.grid[row, col] == self.target
    
    #return true if cell is an obstacle (its value is 1)
    def is_obstacle_cell(self, row: int, col: int) -> bool:
        return self.grid[row, col] == self.obstacle
    
    #return true if current cell is in bounds of the grid
    def is_in_bounds(self, row: int, col: int) -> bool:
        return (0<= row < self.n_rows) and (0<= col < self.n_cols)
    
    
    #we define a function that tracks the trace the agent takes for a single step and checks if this was a valid move
    '''We model movement as straight-line traversal determined by discrete velocity and check intermediate grid cells for collisions'''
    def trace(self, starting_point: Tuple[int, int], end_point: Tuple[int, int]) -> Dict[str, Any]:
        row0, col0 = starting_point
        row1, col1 = end_point
        distance_row = row1 - row0
        distance_col = col1 - col0
        #defines how many cells we need to check along the path
        steps_taken = int(max(abs(distance_row), abs(distance_col)))
        
        #we divine a list of points that the agent passes through
        visited: List[Tuple[int, int]] = []
        
        #now we fill this list with all cells the agent visited along this current step
        if steps_taken == 0:
            visited.append((row0, col0))
        else:
            # Calculate intermediate points
            for s in range(1, steps_taken +1):
                row_visited = int(round(row0 + distance_row * s / steps_taken))
                col_visited = int(round(col0 + distance_col * s / steps_taken))
                #ensure we only add unique cells if it is not already in the visited lists last cell
                if visited == [] or visited[-1] != (row_visited, col_visited):
                    visited.append((row_visited, col_visited))
                
        #now that we have all cells visited we check if any of them are bounds, obstacles, or the target
        for row_visited, col_visited in visited:
            #returns for boundary checks
            if not self.is_in_bounds(row_visited, col_visited):
                return{"visited": visited, "collision": True, "hit_target": False, "reason": "wall", "end_position": starting_point} #return to ponint before collision
            if self.is_obstacle_cell(row_visited, col_visited):
                return{"visited": visited, "collision": True, "hit_target": False, "reason": "obstacle", "end_position": starting_point} #return to point before collision
            if self.is_target_cell(row_visited, col_visited):
                return{"visited": visited, "collision": False, "hit_target": True, "reason": "target", "end_position": (row_visited, col_visited)} #return position of target hit

        return{"visited": visited, "collision": False, "hit_target": False, "reason": "movement ok", "end_position": end_point} #no collision or target hit, step was valid