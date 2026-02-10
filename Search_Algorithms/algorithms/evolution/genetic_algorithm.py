import numpy as np
from algorithms.base.optimizer_base import Optimizer

class GeneticAlgorithm(Optimizer):
    def run(self):
        # 1. Load configuration
        pop_size = self.config.get("pop_size", 50)
        mutation_rate = self.config.get("mutation_rate", 0.1)
        crossover_rate = self.config.get("crossover_rate", 0.9)
        elite_size = self.config.get("elite_size", 2)
        max_iters = self.config.get("max_iters", 1000)
        
        # Validate elite_size
        if elite_size >= pop_size:
            elite_size = max(1, pop_size // 10)  # Default to 10% of population

        # 2. Initialize Population
        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()
        
        # Tạo quần thể ngẫu nhiên [pop_size, dim]
        population = np.random.uniform(low, high, (pop_size, dim))
        fitness = np.array([self.evaluate(ind) for ind in population])
        
        # Xử lý trường hợp evaluate trả về None (hết ngân sách đánh giá)
        if any(f is None for f in fitness):
            return self._build_result()

        # Update global best
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        # 3. Main Loop
        for iteration in range(max_iters):
            if self.evaluations >= self.max_evals:
                break

            # --- Elitism: Giữ lại các cá thể tốt nhất ---
            sorted_indices = np.argsort(fitness)
            new_population = [population[i].copy() for i in sorted_indices[:elite_size]]

            # --- Generate new offspring ---
            while len(new_population) < pop_size:
                if self.evaluations >= self.max_evals:
                    break

                # Selection (Tournament)
                p1 = self._tournament_selection(population, fitness)
                p2 = self._tournament_selection(population, fitness)

                # Crossover
                if np.random.rand() < crossover_rate:
                    c1, c2 = self._arithmetic_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                self._gaussian_mutation(c1, low, high, mutation_rate)
                self._gaussian_mutation(c2, low, high, mutation_rate)

                # Add to new population (Evaluate later to batch processing if needed, but here simple)
                new_population.append(c1)
                if len(new_population) < pop_size:
                    new_population.append(c2)

            # Cắt dư nếu while loop chạy lố
            population = np.array(new_population[:pop_size])
            
            # Evaluate new population
            fitness = []
            for ind in population:
                f = self.evaluate(ind)
                if f is None: # Hết quota
                    f = float('inf') 
                fitness.append(f)
            fitness = np.array(fitness)

            # Update Best
            min_fit_idx = np.argmin(fitness)
            if fitness[min_fit_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fit_idx]
                self.best_solution = population[min_fit_idx].copy()

            self.history.append(self.best_fitness)

        return self._build_result()

    def _tournament_selection(self, population, fitness, k=3):
        indices = np.random.choice(len(population), k, replace=False)
        best_idx = indices[np.argmin(fitness[indices])]
        return population[best_idx]

    def _arithmetic_crossover(self, p1, p2):
        alpha = np.random.rand()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return c1, c2

    def _gaussian_mutation(self, individual, low, high, mutation_rate):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                sigma = (high - low) * 0.1 # Độ lệch chuẩn = 10% phạm vi
                individual[i] += np.random.normal(0, sigma)
                # Clip bounds
                individual[i] = np.clip(individual[i], low, high)