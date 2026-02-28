import numpy as np
from algorithms.base.optimizer_base import Optimizer


class ABC_Knapsack(Optimizer):

    def run(self):
        pop_size = self.config.get("pop_size", 50)
        limit = self.config.get("limit", 50)
        max_iters = self.config.get("max_iters", 500)
        penalty_coef = self.config.get("penalty_coef", 1000)

        n = self.problem.n

        # 1. Initialize binary population
        population = np.random.randint(0, 2, (pop_size, n))
        fitness = np.array([
            self._fitness(ind, penalty_coef)
            for ind in population
        ])

        trial_counters = np.zeros(pop_size)

        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        # 2. Main loop
        for _ in range(max_iters):

            # -------- Employed Bees --------
            for i in range(pop_size):

                k = i
                while k == i:
                    k = np.random.randint(0, pop_size)

                new_sol = self._binary_neighbor(population[i], population[k])
                f_new = self._fitness(new_sol, penalty_coef)

                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # -------- Onlooker Bees --------
            prob = self._selection_prob(fitness)

            for _ in range(pop_size):
                i = self._roulette_wheel_selection(prob)

                k = i
                while k == i:
                    k = np.random.randint(0, pop_size)

                new_sol = self._binary_neighbor(population[i], population[k])
                f_new = self._fitness(new_sol, penalty_coef)

                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # -------- Scout Bees --------
            max_trials_idx = np.argmax(trial_counters)
            if trial_counters[max_trials_idx] > limit:
                population[max_trials_idx] = np.random.randint(0, 2, n)
                fitness[max_trials_idx] = self._fitness(
                    population[max_trials_idx], penalty_coef
                )
                trial_counters[max_trials_idx] = 0

            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = population[best_idx].copy()

            self.history.append(self.best_fitness)

        # Convert back to maximize value
        self.solution = self.best_solution.tolist()
        self.cost = -self._total_value(self.best_solution)

        return self._build_result()

    # --------------------------------------------------
    # Fitness
    # --------------------------------------------------
    def _fitness(self, individual, penalty_coef):
        total_weight = 0
        total_value = 0

        for bit, item in zip(individual, self.problem.items):
            _, w, v = item
            if bit == 1:
                total_weight += w
                total_value += v

        if total_weight > self.problem.capacity:
            penalty = penalty_coef * (total_weight - self.problem.capacity)
        else:
            penalty = 0

        return -total_value + penalty

    def _total_value(self, individual):
        total_value = 0
        for bit, item in zip(individual, self.problem.items):
            if bit == 1:
                total_value += item[2]
        return total_value

    # --------------------------------------------------
    # Binary neighbor
    # --------------------------------------------------
    def _binary_neighbor(self, xi, xk):
        new_sol = xi.copy()

        # Flip 1 random bit influenced by xk
        idx = np.random.randint(0, len(xi))

        if xi[idx] == xk[idx]:
            new_sol[idx] = 1 - xi[idx]
        else:
            new_sol[idx] = xk[idx]

        return new_sol

    # --------------------------------------------------
    # Selection probability
    # --------------------------------------------------
    def _selection_prob(self, fitness):
        # convert minimize → higher probability better
        fit = 1 / (1 + fitness - np.min(fitness))
        return fit / np.sum(fit)

    def _roulette_wheel_selection(self, prob):
        r = np.random.rand()
        cumsum = np.cumsum(prob)
        return np.searchsorted(cumsum, r)