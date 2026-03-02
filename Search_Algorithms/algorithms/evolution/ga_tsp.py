import numpy as np
import time
from algorithms.base.optimizer_base import Optimizer


class GeneticAlgorithmTSP(Optimizer):

    def run(self):
        self.start_time = time.time()

        # 1. Config
        pop_size = self.config.get("pop_size", 100)
        mutation_rate = self.config.get("mutation_rate", 0.2)
        crossover_rate = self.config.get("crossover_rate", 0.9)
        elite_size = self.config.get("elite_size", 2)
        max_iters = self.config.get("max_iters", 500)

        if elite_size >= pop_size:
            elite_size = max(1, pop_size // 10)

        n = self.problem.get_dimension()
        dist_matrix = self.problem.matrix

        # 2. Initialize population (permutations)
        population = np.array(
            [np.random.permutation(n) for _ in range(pop_size)]
        )

        fitness = np.array([self._tour_cost(ind, dist_matrix) for ind in population])

        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        # 3. Main loop
        for _ in range(max_iters):

            new_population = []

            # --- Elitism ---
            sorted_indices = np.argsort(fitness)
            for i in sorted_indices[:elite_size]:
                new_population.append(population[i].copy())

            # --- Generate offspring ---
            while len(new_population) < pop_size:

                p1 = self._tournament_selection(population, fitness)
                p2 = self._tournament_selection(population, fitness)

                if np.random.rand() < crossover_rate:
                    c1, c2 = self._order_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                if np.random.rand() < mutation_rate:
                    self._swap_mutation(c1)

                if np.random.rand() < mutation_rate:
                    self._swap_mutation(c2)

                new_population.append(c1)
                if len(new_population) < pop_size:
                    new_population.append(c2)

            population = np.array(new_population)

            # Evaluate
            fitness = np.array([self._tour_cost(ind, dist_matrix) for ind in population])

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_solution = population[min_idx].copy()

            self.history.append(self.best_fitness)

        # Set final result
        self.solution = self.best_solution.tolist()
        self.cost = float(self.best_fitness)

        return self._build_result()

    # --------------------------------------------------
    # Fitness: total tour cost (return to start)
    # --------------------------------------------------
    def _tour_cost(self, tour, dist_matrix):
        cost = 0
        for i in range(len(tour) - 1):
            cost += dist_matrix[tour[i]][tour[i + 1]]

        # return to start
        cost += dist_matrix[tour[-1]][tour[0]]
        return cost

    # --------------------------------------------------
    # Selection
    # --------------------------------------------------
    def _tournament_selection(self, population, fitness, k=3):
        indices = np.random.choice(len(population), k, replace=False)
        best = indices[np.argmin(fitness[indices])]
        return population[best]

    # --------------------------------------------------
    # Order Crossover (OX)
    # --------------------------------------------------
    def _order_crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(np.random.choice(range(size), 2, replace=False))

        child1 = [-1] * size
        child2 = [-1] * size

        # Copy slice
        child1[a:b] = p1[a:b]
        child2[a:b] = p2[a:b]

        # Fill remaining genes
        def fill(child, parent):
            pos = b
            for gene in parent:
                if gene not in child:
                    if pos >= size:
                        pos = 0
                    child[pos] = gene
                    pos += 1
            return np.array(child)

        child1 = fill(child1, p2)
        child2 = fill(child2, p1)

        return child1, child2

    # --------------------------------------------------
    # Swap Mutation
    # --------------------------------------------------
    def _swap_mutation(self, individual):
        i, j = np.random.choice(len(individual), 2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]