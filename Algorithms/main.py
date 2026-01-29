import yaml
from problems.discrete.grid_pathfinding import GridPathfinding
from algorithms.classical.bfs import BFS
from algorithms.classical.dfs import DFS
from algorithms.classical.ucs import UCS
from algorithms.classical.greedy import Greedy
from algorithms.classical.astar import AStar
from experiment.experiment_runner import ExperimentRunner
from experiment.logger import save_summary_txt

with open("config/classical.yaml") as f:
    config = yaml.safe_load(f)

grid = [
    [0,0,0,0],
    [1,1,0,1],
    [0,0,0,0],
    [0,1,1,0]
]

problem = GridPathfinding(
    grid,
    (0,0),
    (3,3),
    diagonal=config["problem"]["grid_pathfinding"]["diagonal"]
)

algorithms = {
    "BFS": BFS,
    "DFS": DFS,
    "UCS": UCS,
    "Greedy": Greedy,
    "AStar": AStar
}

for name, Algo in algorithms.items():
    runner = ExperimentRunner(
        Algo,
        problem,
        config["algorithms"][name],
        runs=config["experiment"]["runs"]
    )
    summary = runner.run()
    # Sử dụng đường dẫn tuyệt đối dựa trên thư mục của file main.py để đảm bảo tính nhất quán
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "summary_results", "grid_search.txt")
    save_summary_txt(name, summary, file_path)
    print(name, summary)
