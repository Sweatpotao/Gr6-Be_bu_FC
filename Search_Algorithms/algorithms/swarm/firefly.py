import numpy as np
from algorithms.base.optimizer_base import Optimizer

class FireflyAlgorithm(Optimizer):
    def run(self):
        # 1. Config
        pop_size = self.config.get("pop_size", 40)
        alpha = self.config.get("alpha", 0.2)      # Randomness
        beta0 = self.config.get("beta0", 1.0)      # Attractiveness at r=0
        gamma = self.config.get("gamma", 1.0)      # Absorption coefficient
        max_iters = self.config.get("max_iters", 1000)

        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()
        scale = high - low

        # 2. Init
        fireflies = np.random.uniform(low, high, (pop_size, dim))
        light_intensity = np.array([self.evaluate(ind) for ind in fireflies])
        
        if any(f is None for f in light_intensity): return self._build_result()
        
        # Best
        best_idx = np.argmin(light_intensity)
        self.best_solution = fireflies[best_idx].copy()
        self.best_fitness = light_intensity[best_idx]
        self.history.append(self.best_fitness)

        # 3. Loop
        for iteration in range(max_iters):
            if self.evaluations >= self.max_evals: break

            # Giảm alpha dần theo thời gian (tùy chọn, giúp hội tụ)
            # alpha_t = alpha * (0.97 ** iteration) 
            
            # Loop đôi: so sánh mọi cặp đom đóm
            for i in range(pop_size):
                for j in range(pop_size):
                    if self.evaluations >= self.max_evals: break
                    
                    # Nếu j sáng hơn i (fitness nhỏ hơn vì minimize) -> i di chuyển về j
                    if light_intensity[j] < light_intensity[i]:
                        # Tính khoảng cách Euclidean
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        
                        # Tính độ hấp dẫn: beta = beta0 * exp(-gamma * r^2)
                        beta = beta0 * np.exp(-gamma * (r ** 2))
                        
                        # Bước di chuyển
                        # x_i = x_i + beta*(x_j - x_i) + alpha * (rand - 0.5)
                        noise = alpha * (np.random.rand(dim) - 0.5) * scale
                        
                        new_pos = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + noise
                        new_pos = np.clip(new_pos, low, high)
                        
                        f_new = self.evaluate(new_pos)
                        if f_new is not None:
                             # FA gốc: luôn di chuyển. Có thể dùng greedy update.
                             # Ở đây dùng greedy update để đảm bảo ko mất nghiệm tốt
                             if f_new < light_intensity[i]:
                                 fireflies[i] = new_pos
                                 light_intensity[i] = f_new

            # Update Global Best
            min_idx = np.argmin(light_intensity)
            if light_intensity[min_idx] < self.best_fitness:
                self.best_fitness = light_intensity[min_idx]
                self.best_solution = fireflies[min_idx].copy()
            
            self.history.append(self.best_fitness)

        return self._build_result()