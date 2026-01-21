import numpy as np

# Grid 1: simple (few obstacles)
grid_easy = np.array([
    [2, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3],
])

# Grid 2: medium difficulty
grid_medium = np.array([
    [2, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 3],
])

# Grid 3: hard (many obstacles)
grid_hard = np.array([
    [2, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 3],
])
