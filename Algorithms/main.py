from experiment.run_experiment import run_experiment
from experiment.logger import save_summary_txt

if __name__ == "__main__":
    results = run_experiment("config/grid_pathfinding.yaml")

    for algo, summary in results.items():
        save_summary_txt(
            algo,
            summary,
            "data/summary_results/grid_pathfinding.txt"
        )
        print(algo, summary)
