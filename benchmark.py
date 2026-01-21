import time
import numpy as np

from grid import env
from grids import grid_easy, grid_medium, grid_hard
from montecarlo import mc_control_train
from policy_eval import greedy_path


def obstacle_count(grid: np.ndarray) -> int:
    return int(np.sum(grid == 1))


def run_one(
    grid: np.ndarray,
    name: str,
    max_steps: int,
    seed: int = 42
):

    e = env(grid, seed=seed)

    start_time = time.perf_counter()
    Q, N = mc_control_train(
        e,
        num_episodes=15000,   # you can change later
        epsilon=0.2,
        gamma=1.0,
        max_steps=max_steps,
        seed=0
    )
    end_time = time.perf_counter()

    # evaluate greedy path length (quality indicator)
    e_eval = env(grid, seed=seed)
    path = greedy_path(e_eval, Q, max_steps=max_steps)


    return {
        "grid": name,
        "rows": grid.shape[0],
        "cols": grid.shape[1],
        "obstacles": obstacle_count(grid),
        "train_seconds": end_time - start_time,
        "q_size": len(Q),
        "greedy_path_len": len(path),
        "reached_target": int(grid[path[-1][0], path[-1][1]] == 3),
    }


if __name__ == "__main__":
    results = []
    MAX_STEPS = 800
    results.append(run_one(grid_easy, "easy", MAX_STEPS))
    results.append(run_one(grid_medium, "medium", MAX_STEPS))
    results.append(run_one(grid_hard, "hard", MAX_STEPS))


    print("\n=== Benchmark Results ===")
    for r in results:
        print(r)

    # Save as CSV (simple, no pandas needed)
    with open("benchmark_results.csv", "w", encoding="utf-8") as f:
        header = results[0].keys()
        f.write(",".join(header) + "\n")
        for r in results:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    print("\nSaved: benchmark_results.csv")
