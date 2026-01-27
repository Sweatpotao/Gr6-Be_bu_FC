import time
import numpy as np

class ExperimentRunner:
    def __init__(self, algorithm_class, problem, algo_config, runs=1):
        self.algorithm_class = algorithm_class
        self.problem = problem
        self.algo_config = algo_config
        self.runs = runs

    def run(self):
        results = []
        times = []

        for _ in range(self.runs):
            algo = self.algorithm_class(self.problem, self.algo_config)

            start = time.perf_counter()
            result = algo.search()
            end = time.perf_counter()

            result["runtime"] = end - start
            results.append(result)
            times.append(result["runtime"])

        costs = [r["cost"] for r in results]
        best_idx = int(np.argmin(costs))

        return {
            "runs": self.runs,
            "best_cost": min(costs),
            "mean_cost": np.mean(costs),
            "std_cost": np.std(costs),
            "mean_time": np.mean(times),
            "example_result": results[best_idx],
            "all_results": results
        }
