import numpy as np
from algorithms.base.optimizer_base import Optimizer

class ACO(Optimizer):
    """
    Ant Colony Optimization for Continuous Domains (ACOR).
    Dựa trên kho lưu trữ giải pháp (Archive) và Gaussian Kernel PDF.
    """
    def run(self):
        # 1. Config
        # pop_size ở đây đóng vai trò là kích thước Archive (k)
        archive_size = self.config.get("archive_size", 50) 
        m = self.config.get("sample_size", 30) # Số lượng kiến sinh ra mỗi vòng
        q = self.config.get("q", 0.1) # Tham số trọng số (locality of search)
        xi = self.config.get("xi", 0.85) # Tốc độ hội tụ (giống pheromone evaporation)
        max_iters = self.config.get("max_iters", 1000)

        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()

        # 2. Init Archive
        # Archive lưu (solution, fitness)
        population = np.random.uniform(low, high, (archive_size, dim))
        fitness = np.array([self.evaluate(ind) for ind in population])
        
        # Sort archive theo fitness (tốt nhất ở index 0)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        self.best_solution = population[0].copy()
        self.best_fitness = fitness[0]
        self.history.append(self.best_fitness)

        # Tính trọng số w cho archive: w_l = 1 / (q * k * sqrt(2pi)) * exp(...)
        # Công thức đơn giản hóa: w_i propto exp( - (rank-1)^2 / (2q^2k^2) )
        ranks = np.arange(1, archive_size + 1)
        weights = np.exp(-((ranks - 1) ** 2) / (2 * (q * archive_size) ** 2))
        prob = weights / np.sum(weights)

        # 3. Main Loop
        for iteration in range(max_iters):
            if self.evaluations >= self.max_evals: break

            new_solutions = []
            new_fitnesses = []

            # Sinh ra m kiến mới
            for _ in range(m):
                if self.evaluations >= self.max_evals: break

                # Chọn một giải pháp hướng dẫn (guide) từ archive dựa trên prob
                l = self._roulette_wheel_selection(prob)
                guide_sol = population[l]

                # Tính sigma (độ lệch chuẩn) cho bước nhảy Gaussian
                # sigma_i = xi * sum(|x_l - x_r|) / (k-1)
                # Để nhanh, ta tính khoảng cách trung bình từ guide đến các điểm khác
                mean_dist = np.mean(np.abs(population - guide_sol), axis=0)
                sigma = xi * mean_dist
                
                # Sinh giải pháp mới: S_new ~ N(S_guide, sigma)
                new_sol = guide_sol + np.random.normal(0, 1, dim) * sigma
                new_sol = np.clip(new_sol, low, high)

                f = self.evaluate(new_sol)
                if f is None: break

                new_solutions.append(new_sol)
                new_fitnesses.append(f)

                if f < self.best_fitness:
                    self.best_fitness = f
                    self.best_solution = new_sol.copy()

            self.history.append(self.best_fitness)

            # Cập nhật Archive: Gộp cũ + mới rồi giữ lại top k tốt nhất
            if len(new_solutions) > 0:
                combined_pop = np.vstack((population, np.array(new_solutions)))
                combined_fit = np.concatenate((fitness, np.array(new_fitnesses)))
                
                sorted_idx = np.argsort(combined_fit)
                # Cắt lấy top k
                population = combined_pop[sorted_idx[:archive_size]]
                fitness = combined_fit[sorted_idx[:archive_size]]

        return self._build_result()

    def _roulette_wheel_selection(self, prob):
        r = np.random.rand()
        cumsum = np.cumsum(prob)
        return np.searchsorted(cumsum, r)