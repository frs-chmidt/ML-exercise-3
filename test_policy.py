import numpy as np
from grid import env
from montecarlo import mc_control_train
from policy_eval import greedy_path

# Simple grid
grid = np.array([
    [2, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3],
])

e = env(grid, seed=42)

Q, N = mc_control_train(
    e,
    num_episodes=3000,
    epsilon=0.2,
    gamma=1.0,
    max_steps=200,
    seed=0
)

path = greedy_path(e, Q)

print("Greedy path length:", len(path))
print("Greedy path states:")
for s in path:
    print(s)
