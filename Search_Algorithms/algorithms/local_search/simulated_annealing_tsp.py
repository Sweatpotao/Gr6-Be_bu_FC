import numpy as np
import math
import time
from algorithms.base.optimizer_base import Optimizer


class SimulatedAnnealingTSP(Optimizer):

    def run(self):
        self.start_time = time.time()

        T = self.config.get("initial_temp", 1000)
        alpha = self.config.get("cooling_rate", 0.995)
        max_iters = self.config.get("max_iters", 5000)
        use_two_opt = self.config.get("two_opt", True)

        n = self.problem.get_dimension()
        dist_matrix = self.problem.matrix

        # Initial solution
        current = np.random.permutation(n)
        current_f = self._tour_cost(current, dist_matrix)

        self.best_solution = current.copy()
        self.best_fitness = current_f
        self.history.append(current_f)

        # Main loop
        for _ in range(max_iters):

            if self._check_timeout():
                break

            # Generate neighbor
            if use_two_opt:
                neighbor = self._two_opt_neighbor(current)
            else:
                neighbor = self._swap_neighbor(current)

            f = self._tour_cost(neighbor, dist_matrix)
            delta = f - current_f

            # Metropolis criterion
            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                current = neighbor
                current_f = f

                if current_f < self.best_fitness:
                    self.best_solution = current.copy()
                    self.best_fitness = current_f

            self.history.append(self.best_fitness)

            # Cooling
            T *= alpha
            if T < 1e-8:
                break

        self.solution = self.best_solution.tolist()
        self.cost = float(self.best_fitness)

        return self._build_result()

    # ----------------------------------------
    # Cost
    # ----------------------------------------
    def _tour_cost(self, tour, dist_matrix):
        cost = 0
        for i in range(len(tour) - 1):
            cost += dist_matrix[tour[i]][tour[i + 1]]
        cost += dist_matrix[tour[-1]][tour[0]]
        return cost

    # ----------------------------------------
    # Swap neighbor
    # ----------------------------------------
    def _swap_neighbor(self, tour):
        neighbor = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    # ----------------------------------------
    # 2-opt neighbor (recommended)
    # ----------------------------------------
    def _two_opt_neighbor(self, tour):
        neighbor = tour.copy()
        i, j = sorted(np.random.choice(len(tour), 2, replace=False))
        neighbor[i:j] = neighbor[i:j][::-1]
        return neighbor