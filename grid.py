#here we will create the grid (environments with different complexities)

class env:
    def __init__(self, grid_size, start, target, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.target = target
        self.obstacles = obstacles


