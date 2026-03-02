import numpy as np
import time
from algorithms.base.optimizer_base import Optimizer


class HillClimbingTSP(Optimizer):

    def run(self):
        self.start_time = time.time()

        max_iters = self.config.get("max_iters", 1000)
        n_neighbors = self.config.get("n_neighbors", 50)
        use_two_opt = self.config.get("two_opt", True)

        n = self.problem.get_dimension()
        dist_matrix = self.problem.matrix

        # 1️⃣ Initial solution (random permutation)
        current = np.random.permutation(n)
        current_f = self._tour_cost(current, dist_matrix)

        self.best_solution = current.copy()
        self.best_fitness = current_f
        self.history.append(current_f)

        # 2️⃣ Main loop
        for _ in range(max_iters):

            if self._check_timeout():
                break

            improved = False

            for _ in range(n_neighbors):

                if use_two_opt:
                    neighbor = self._two_opt_neighbor(current)
                else:
                    neighbor = self._swap_neighbor(current)

                f = self._tour_cost(neighbor, dist_matrix)

                if f < current_f:
                    current = neighbor
                    current_f = f
                    improved = True

            self.history.append(current_f)

            if not improved:
                break

        self.solution = current.tolist()
        self.cost = float(current_f)

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
    # 2-opt neighbor
    # ----------------------------------------
    def _two_opt_neighbor(self, tour):
        neighbor = tour.copy()
        i, j = sorted(np.random.choice(len(tour), 2, replace=False))
        neighbor[i:j] = neighbor[i:j][::-1]
        return neighbor