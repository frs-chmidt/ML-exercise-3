import numpy as np
from grid import env
from montecarlo import mc_control_train

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
    num_episodes=200,   # small test
    epsilon=0.3,
    gamma=1.0,
    max_steps=200,
    seed=0
)

print("Training finished.")
print("Number of learned Q entries:", len(Q))
print("Number of visited (s,a) entries:", len(N))

# Show 5 sample learned entries
print("\nSample Q values:")
for i, (k, v) in enumerate(Q.items()):
    if i == 5:
        break
    print(k, "->", v, "visits:", N[k])
