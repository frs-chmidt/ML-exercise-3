import numpy as np
from grid import env
from episode import generate_episode_random

# Same simple grid as before
grid = np.array([
    [2, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3],
])

e = env(grid, seed=42)

episode = generate_episode_random(e, max_steps=50)

print("Episode length:", len(episode))
print("First 5 transitions:")
for i, (s, a, r) in enumerate(episode[:5]):
    print(i, "state=", s, "action=", a, "reward=", r)

from episode import compute_returns

returns = compute_returns(episode, gamma=1.0)
print("First 5 returns:", returns[:5])
print("Last 5 returns:", returns[-5:])
