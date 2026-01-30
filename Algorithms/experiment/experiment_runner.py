import time
import numpy as np

class ExperimentRunner:
    def __init__(self, algorithm_class, problem, algo_config, runs=1):
        self.algorithm_class = algorithm_class
        self.problem = problem
        self.algo_config = algo_config
        self.runs = runs

    def run(self):
<<<<<<< HEAD
        results = []
        times = []

        for _ in range(self.runs):
            # Create a fresh copy of the problem for each run
            problem_instance = self.problem.clone() if hasattr(self.problem, 'clone') else self.problem
            
            algo = self.algorithm_class(problem_instance, self.algo_config)

            start = time.perf_counter()
            result = algo.search()
            end = time.perf_counter()

            result["runtime"] = end - start
            results.append(result)
            times.append(result["runtime"])

        # Handle case where some runs may not find a solution (cost is None)
        valid_costs = [r["cost"] for r in results if r["cost"] is not None]
        if not valid_costs:
            # No solution found in any run
            return {
                "runs": self.runs,
                "best_cost": None,
                "mean_cost": None,
                "std_cost": None,
                "mean_time": np.mean(times),
                "example_result": results[0] if results else None,
                "all_results": results,
                "success_rate": 0.0
            }

        costs = [r["cost"] if r["cost"] is not None else float('inf') for r in results]
        best_idx = int(np.argmin(costs))

        return {
            "runs": self.runs,
            "best_cost": min(valid_costs),
            "mean_cost": np.mean(valid_costs),
            "std_cost": np.std(valid_costs) if len(valid_costs) > 1 else 0.0,
            "mean_time": np.mean(times),
            "example_result": results[best_idx],
            "all_results": results,
            "success_rate": len(valid_costs) / self.runs
=======
        scores, times, efforts, histories, successes = [], [], [], [], []

        for _ in range(self.runs):
            prob = self.problem.clone() if hasattr(self.problem, "clone") else self.problem
            algo = self.algorithm_class(prob, self.algo_config)

            start = time.perf_counter()
            result = algo.search() if hasattr(algo, "search") else algo.run()
            runtime = time.perf_counter() - start

            result["runtime"] = runtime

            if result["found"]:
                scores.append(result["final_score"])
                successes.append(1)
            else:
                successes.append(0)

            times.append(runtime)

            # effort
            if "nodes_expanded" in result:
                efforts.append(result["nodes_expanded"])
            elif "evaluations" in result:
                efforts.append(result["evaluations"])

            histories.append(result.get("history"))

        return {
            "runs": self.runs,

            # Solution quality
            "best_score": min(scores) if scores else None,
            "mean_score": np.mean(scores) if scores else None,
            "std_score": np.std(scores) if len(scores) > 1 else 0.0,

            # Performance
            "mean_time": np.mean(times),
            "mean_effort": np.mean(efforts),

            # Reliability
            "success_rate": np.mean(successes),

            # Convergence
            "histories": histories
>>>>>>> b48bc3d2c962571356227dca0eba443e675d34b7
        }
