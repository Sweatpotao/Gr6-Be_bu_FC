import time
import numpy as np

class ExperimentRunner:
    def __init__(self, algorithm_class, problem, algo_config, runs=1):
        self.algorithm_class = algorithm_class
        self.problem = problem
        self.algo_config = algo_config
        self.runs = runs

    def run(self):
        scores, times, efforts, histories, successes, timeouts = [], [], [], [], [], []

        for _ in range(self.runs):
            prob = self.problem.clone() if hasattr(self.problem, "clone") else self.problem
            algo = self.algorithm_class(prob, self.algo_config)

            start = time.perf_counter()
            result = algo.search() if hasattr(algo, "search") else algo.run()
            runtime = time.perf_counter() - start

            result["runtime"] = runtime

            if result.get("found", False):
                score = result.get("final_score")
                if score is not None:
                    scores.append(score)
                    # Phân biệt bài toán Liên tục & Rời rạc
                    if "evaluations" in result:
                        if score < 4:  # Epsilon = 4 vì best score của nhiều thuật toán chỉ ở mức 1.0-4.0 (không chạm đáy tuyệt đối)
                            successes.append(1)
                        else:
                            successes.append(0) # Kẹt ở cực trị địa phương
                    else:
                        successes.append(1)
                else:
                    successes.append(0)
            else:
                successes.append(0)

            # Theo dõi timeout
            if result.get("timeout", False):
                timeouts.append(1)
            else:
                timeouts.append(0)

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

            # Timeout tracking
            "timeout_count": sum(timeouts),

            # Convergence
            "histories": histories
        }
