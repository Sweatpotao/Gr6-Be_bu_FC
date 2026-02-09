import numpy as np
from algorithms.base.optimizer_base import Optimizer

class TLBO(Optimizer):
    def run(self):
        # 1. Config
        pop_size = self.config.get("pop_size", 30)
        max_iters = self.config.get("max_iters", 1000)

        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()

        # 2. Init Class
        population = np.random.uniform(low, high, (pop_size, dim))
        fitness = np.array([self.evaluate(ind) for ind in population])
        
        if any(f is None for f in fitness): return self._build_result()

        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        # 3. Loop
        for iteration in range(max_iters):
            if self.evaluations >= self.max_evals: break

            # Xác định Teacher (người giỏi nhất lớp)
            teacher_idx = np.argmin(fitness)
            teacher = population[teacher_idx]
            
            # Tính Mean của lớp
            mean_pop = np.mean(population, axis=0)

            # --- Teacher Phase ---
            for i in range(pop_size):
                if self.evaluations >= self.max_evals: break
                
                # Teaching factor TF có thể là 1 hoặc 2
                tf = np.random.randint(1, 3) 
                r = np.random.rand(dim)
                
                # Học sinh cố gắng học theo thầy
                diff = r * (teacher - tf * mean_pop)
                new_sol = population[i] + diff
                new_sol = np.clip(new_sol, low, high)
                
                f_new = self.evaluate(new_sol)
                if f_new is None: break

                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new

            # --- Learner Phase ---
            for i in range(pop_size):
                if self.evaluations >= self.max_evals: break
                
                # Chọn bạn học ngẫu nhiên j khác i
                idxs = [idx for idx in range(pop_size) if idx != i]
                j = np.random.choice(idxs)
                
                r = np.random.rand(dim)
                
                # Nếu bạn j giỏi hơn -> học theo j, ngược lại -> tránh xa
                if fitness[j] < fitness[i]:
                    step = population[j] - population[i]
                else:
                    step = population[i] - population[j]
                
                new_sol = population[i] + r * step
                new_sol = np.clip(new_sol, low, high)
                
                f_new = self.evaluate(new_sol)
                if f_new is None: break

                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new

            # Update Global Best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_solution = population[min_idx].copy()
            
            self.history.append(self.best_fitness)

        return self._build_result()