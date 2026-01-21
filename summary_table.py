import csv

def read_csv(filename="benchmark_results.csv"):
    with open(filename, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

if __name__ == "__main__":
    rows = read_csv()

    # sort by obstacles
    rows = sorted(rows, key=lambda r: int(r["obstacles"]))

    print("\n=== Final Benchmark Summary ===")
    print(f"{'grid':<8} {'size':<6} {'obs':<4} {'time(s)':<10} {'Q':<6} {'path_len':<9} {'reached':<7}")
    print("-" * 60)

    for r in rows:
        grid = r["grid"]
        size = f'{r["rows"]}x{r["cols"]}'
        obs = int(r["obstacles"])
        t = float(r["train_seconds"])
        q = int(r["q_size"])
        pl = int(r["greedy_path_len"])
        reached = "yes" if int(r["reached_target"]) == 1 else "no"
        print(f"{grid:<8} {size:<6} {obs:<4} {t:<10.3f} {q:<6} {pl:<9} {reached:<7}")
