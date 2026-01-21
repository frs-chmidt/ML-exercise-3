import numpy as np
import matplotlib.pyplot as plt

from grid import env
from grids import grid_easy, grid_medium, grid_hard
from montecarlo import mc_control_train
from policy_eval import greedy_path


def plot_grid_and_path(grid: np.ndarray, path, title: str, filename: str):
    """
    Plot grid (0 free, 1 obstacle, 2 start, 3 target) and overlay greedy path.
    """
    plt.figure()
    plt.imshow(grid, interpolation="nearest")

    # path contains states (row, col, vx, vy); we plot (row,col)
    rows = [s[0] for s in path]
    cols = [s[1] for s in path]

    plt.plot(cols, rows, marker="o")  # x=col, y=row
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("col")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")


def train_and_plot(grid: np.ndarray, name: str, episodes: int, max_steps: int):
    e = env(grid, seed=42)
    Q, _N = mc_control_train(
        e,
        num_episodes=episodes,
        epsilon=0.2,      # epsilon decay is already inside  mc_control_train
        gamma=1.0,
        max_steps=max_steps,
        seed=0
    )

    e_eval = env(grid, seed=42)
    path = greedy_path(e_eval, Q, max_steps=max_steps)

    reached = int(grid[path[-1][0], path[-1][1]] == 3)
    title = f"{name.capitalize()} grid – Greedy path (reached={reached}, len={len(path)})"
    filename = f"greedy_path_{name}.png"

    plot_grid_and_path(grid, path, title, filename)


if __name__ == "__main__":
    # Keep these consistent with benchmark.py
    MAX_STEPS = 800
    EPISODES = 15000

    train_and_plot(grid_easy, "easy", episodes=EPISODES, max_steps=MAX_STEPS)
    train_and_plot(grid_medium, "medium", episodes=EPISODES, max_steps=MAX_STEPS)
    train_and_plot(grid_hard, "hard", episodes=EPISODES, max_steps=MAX_STEPS)
