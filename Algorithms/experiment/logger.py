import os
from datetime import datetime

<<<<<<< HEAD
def save_summary_txt(algo_name, summary, filename="summary_results.txt"):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(filename, "a", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Experiment time: {datetime.now()}\n")
        f.write(f"Algorithm: {algo_name}\n")

        f.write(f"Runs: {summary['runs']}\n")
        
        # Handle case where no solution was found
        if summary.get('best_cost') is None:
            f.write("Status: No solution found\n")
            f.write(f"Success rate: {summary.get('success_rate', 0.0):.2%}\n")
        else:
            f.write(f"Best cost: {summary['best_cost']:.4f}\n")
            f.write(f"Mean cost: {summary['mean_cost']:.4f}\n")
            f.write(f"Std cost: {summary['std_cost']:.4f}\n")
            if 'success_rate' in summary:
                f.write(f"Success rate: {summary['success_rate']:.2%}\n")
        
        f.write(f"Mean time: {summary['mean_time']:.6e} seconds\n")

        ex = summary["example_result"]
        if ex:
            f.write(f"Nodes expanded: {ex['nodes_expanded']}\n")
            if ex.get('found'):
                f.write(f"Path: {ex['solution']}\n")
            else:
                f.write("Path: No solution found\n")
        f.write("\n")
=======
def save_summary_txt(algo_name, summary, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Experiment time : {datetime.now()}\n")
        f.write(f"Algorithm       : {algo_name}\n")
        f.write(f"Runs            : {summary['runs']}\n")

        if summary["best_score"] is None:
            f.write("Status          : No solution found\n")
        else:
            f.write(f"Best score      : {summary['best_score']:.6f}\n")
            f.write(f"Mean score      : {summary['mean_score']:.6f}\n")
            f.write(f"Std score       : {summary['std_score']:.6f}\n")

        f.write(f"Success rate    : {summary['success_rate']:.2%}\n")
        f.write(f"Mean time (s)   : {summary['mean_time']:.6e}\n")
        f.write(f"Mean effort     : {summary['mean_effort']:.2f}\n\n")
>>>>>>> b48bc3d2c962571356227dca0eba443e675d34b7
