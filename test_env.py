import numpy as np
from grid import env

# Simple test grid
# 0 = free, 1 = obstacle, 2 = start, 3 = target
grid = np.array([
    [2, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3],
])

# Create environment
e = env(grid, seed=42)

# Reset environment
state = e.reset()
print("Initial state:", state)

# Take a few steps with a fixed action
for t in range(5):
    action = (1, 0)  # accelerate downward
    state, reward, done, info = e.step(action)

    print(f"Step {t}")
    print("  Action:", action)
    print("  State:", state)
    print("  Reward:", reward)
    print("  Done:", done)
    print("  Info:", info)

    if done:
        print("Target reached!")
        break
