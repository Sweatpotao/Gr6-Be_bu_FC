import numpy as np
from algorithms.base.optimizer_base import Optimizer

class DifferentialEvolution(Optimizer):
    def run(self):
        # Config
        pop_size = self.config.get("pop_size", 50)
        F = self.config.get("F", 0.8)      # Scaling factor (Mutation)
        Cr = self.config.get("Cr", 0.9)    # Crossover probability
        max_iters = self.config.get("max_iters", 1000)

        # Init params
        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()

        # Init population
        population = np.random.uniform(low, high, (pop_size, dim))
        fitness = np.array([self.evaluate(ind) for ind in population])

        if any(f is None for f in fitness): return self._build_result()

        # Find best
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        for iteration in range(max_iters):
            if self.evaluations >= self.max_evals:
                break
            
            # Convergence check: stop if population has converged
            pop_std = np.std(population)
            if pop_std < 1e-10:
                break
            
            new_population = []
            new_fitness = []

            for i in range(pop_size):
                # 1. Mutation: Chọn 3 vector khác nhau và khác i
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Vector đột biến v = a + F * (b - c)
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, low, high)

                # 2. Crossover: Tạo trial vector u từ x(i) và mutant
                cross_points = np.random.rand(dim) < Cr
                # Luôn giữ ít nhất 1 chiều từ mutant để đảm bảo biến đổi
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])

                # 3. Selection: Greedy
                f_trial = self.evaluate(trial)
                if f_trial is None: break

                if f_trial < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                    
                    # Update global best ngay lập tức nếu tốt hơn
                    if f_trial < self.best_fitness:
                        self.best_fitness = f_trial
                        self.best_solution = trial.copy()
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            self.history.append(self.best_fitness)

        return self._build_result()