import os
from datetime import datetime

def save_summary_txt(algo_name, summary, filename="summary_results.txt"):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(filename, "a", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Experiment time: {datetime.now()}\n")
        f.write(f"Algorithm: {algo_name}\n")

        f.write(f"Runs: {summary['runs']}\n")
        f.write(f"Best cost: {summary['best_cost']}\n")
        f.write(f"Mean cost: {summary['mean_cost']}\n")
        f.write(f"Std cost: {summary['std_cost']}\n")
        f.write(f"Mean time: {summary['mean_time']:.6e}\n")

        ex = summary["example_result"]
        f.write(f"Nodes expanded: {ex['nodes_expanded']}\n")
        f.write(f"Path: {ex['solution']}\n\n")
