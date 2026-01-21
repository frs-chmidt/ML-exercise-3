import csv
import matplotlib.pyplot as plt

def read_csv(filename="benchmark_results.csv"):
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert numeric fields
            r["obstacles"] = int(r["obstacles"])
            r["train_seconds"] = float(r["train_seconds"])
            r["reached_target"] = int(r["reached_target"])
            rows.append(r)
    return rows

if __name__ == "__main__":
    data = read_csv()

    obstacles = [d["obstacles"] for d in data]
    times = [d["train_seconds"] for d in data]
    labels = [d["grid"] for d in data]

    plt.figure()
    plt.plot(obstacles, times, marker="o")
    for x, y, lab in zip(obstacles, times, labels):
        plt.text(x, y, lab)

    plt.xlabel("Number of obstacles")
    plt.ylabel("Training time (seconds)")
    plt.title("Runtime vs obstacle count (MC Control)")
    plt.tight_layout()
    plt.savefig("runtime_vs_obstacles.png", dpi=200)
    plt.close()

    print("Saved: runtime_vs_obstacles.png")
