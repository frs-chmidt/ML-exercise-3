import csv
import matplotlib.pyplot as plt

def read_csv(filename="benchmark_results.csv"):
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["obstacles"] = int(r["obstacles"])
            r["train_seconds"] = float(r["train_seconds"])
            r["reached_target"] = int(r["reached_target"])
            rows.append(r)
    return rows

if __name__ == "__main__":
    data = read_csv()

    # sort by obstacles so the line is monotonic in x
    data = sorted(data, key=lambda d: d["obstacles"])

    x = [d["obstacles"] for d in data]
    y = [d["train_seconds"] for d in data]
    labels = [d["grid"] for d in data]
    reached = [d["reached_target"] for d in data]

    plt.figure()
    plt.plot(x, y, marker="o")

    # annotate each point with grid name + success/fail
    for xi, yi, lab, ok in zip(x, y, labels, reached):
        status = "ok" if ok == 1 else "FAIL"
        plt.text(xi, yi, f"{lab} ({status})")

    plt.xlabel("Number of obstacles")
    plt.ylabel("Training time (seconds)")
    plt.title("Runtime vs obstacle count (MC Control)")
    plt.tight_layout()
    plt.savefig("runtime_vs_obstacles_v2.png", dpi=200)
    plt.close()

    print("Saved: runtime_vs_obstacles_v2.png")
