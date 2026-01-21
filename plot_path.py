import numpy as np
import matplotlib.pyplot as plt

from grid import env
from montecarlo import mc_control_train
from policy_eval import greedy_path


def plot_grid_and_path(grid: np.ndarray, path, filename: str = "path.png"):
    """
    Plots the grid and overlays the (row,col) positions of the greedy path.
    """
    # show grid (0 free, 1 obstacle, 2 start, 3 target)
    plt.figure()
    plt.imshow(grid, interpolation="nearest")

    # extract positions from states
    rows = [s[0] for s in path]
    cols = [s[1] for s in path]

    # overlay path
    plt.plot(cols, rows, marker="o")  # note: x=col, y=row

    plt.gca().invert_yaxis()  # so row 0 appears at top like matrix indexing
    plt.title("Greedy path after MC Control")
    plt.xlabel("col")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved figure: {filename}")


if __name__ == "__main__":
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
    plot_grid_and_path(grid, path, filename="greedy_path_grid_simple.png")
