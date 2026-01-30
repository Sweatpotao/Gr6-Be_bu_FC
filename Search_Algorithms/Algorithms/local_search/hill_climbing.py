import numpy as np
import time
from algorithms.base.optimizer_base import Optimizer

class HillClimbing(Optimizer):
    def run(self):
        start_time = time.time()

        step_size = self.config.get("step_size", 0.05)
        n_neighbors = self.config.get("n_neighbors", 20)
        max_iters = self.config.get("max_iters", 1000)

        current = self.problem.initial_solution()
        current_f = self.evaluate(current)

        self.best_solution = current
        self.best_fitness = current_f
        self.history.append(current_f)

        for _ in range(max_iters):
            improved = False

            for _ in range(n_neighbors):
                neighbor = current + np.random.uniform(
                    -step_size, step_size, size=current.shape
                )
                low, high = self.problem.get_bounds()
                neighbor = np.clip(neighbor, low, high)

                f = self.evaluate(neighbor)
                if f is None:
                    break

                if f < current_f:
                    current, current_f = neighbor, f
                    improved = True

            self.history.append(current_f)

            if not improved or self.evaluations >= self.max_evals:
                break

        self.best_solution = current
        self.best_fitness = current_f
        self.runtime = time.time() - start_time
        return self._build_result()
