import numpy as np
import math
import time
from algorithms.base.optimizer_base import Optimizer

class SimulatedAnnealing(Optimizer):
    def run(self):
        self.start_time = time.time()

        T = self.config.get("initial_temp", 100)
        alpha = self.config.get("cooling_rate", 0.95)
        max_iters = self.config.get("max_iters", 2000)

        current = self.problem.initial_solution()
        current_f = self.evaluate(current)

        self.best_solution = current
        self.best_fitness = current_f
        self.history.append(current_f)

        for _ in range(max_iters):
            # Kiểm tra timeout
            if self._check_timeout():
                break

            if self.evaluations >= self.max_evals:
                break

            neighbor = current + np.random.normal(0, 0.1, size=current.shape)
            low, high = self.problem.get_bounds()
            neighbor = np.clip(neighbor, low, high)

            f = self.evaluate(neighbor)
            if f is None:
                break

            delta = f - current_f
            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                current, current_f = neighbor, f

            self.history.append(current_f)
            T *= alpha

        self.best_solution = current
        self.best_fitness = current_f
        self.runtime = time.time() - self.start_time
        return self._build_result()
