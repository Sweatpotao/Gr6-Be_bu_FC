import numpy as np
from algorithms.base.optimizer_base import Optimizer


class TLBO_Knapsack(Optimizer):

    def run(self):
        pop_size = self.config.get("pop_size", 40)
        max_iters = self.config.get("max_iters", 500)
        penalty_coef = self.config.get("penalty_coef", 1000)

        n = self.problem.n

        # 1️⃣ Initialize binary population
        population = np.random.randint(0, 2, (pop_size, n))
        fitness = np.array([
            self._fitness(ind, penalty_coef)
            for ind in population
        ])

        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        # 2️⃣ Main Loop
        for _ in range(max_iters):

            # =========================
            # TEACHER PHASE
            # =========================
            teacher_idx = np.argmin(fitness)
            teacher = population[teacher_idx]

            mean_bits = np.round(np.mean(population, axis=0))  # mean → majority bit

            for i in range(pop_size):

                new_sol = population[i].copy()

                for d in range(n):
                    r = np.random.rand()

                    # nếu teacher khác majority → tăng xác suất đổi
                    if teacher[d] != mean_bits[d]:
                        if r < 0.5:
                            new_sol[d] = teacher[d]

                f_new = self._fitness(new_sol, penalty_coef)

                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new

            # =========================
            # LEARNER PHASE
            # =========================
            for i in range(pop_size):

                j = i
                while j == i:
                    j = np.random.randint(0, pop_size)

                new_sol = population[i].copy()

                if fitness[j] < fitness[i]:
                    better = population[j]
                else:
                    better = population[i]

                # học theo nghiệm tốt hơn
                for d in range(n):
                    if np.random.rand() < 0.3:
                        new_sol[d] = better[d]

                f_new = self._fitness(new_sol, penalty_coef)

                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new

            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = population[best_idx].copy()

            self.history.append(self.best_fitness)

        # Convert back to maximize
        self.solution = self.best_solution.tolist()
        self.cost = -self._total_value(self.best_solution)

        return self._build_result()

    # =========================
    # Fitness
    # =========================
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