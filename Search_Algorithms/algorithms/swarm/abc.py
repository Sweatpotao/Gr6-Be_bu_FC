import numpy as np
from algorithms.base.optimizer_base import Optimizer

class ABC(Optimizer):
    def run(self):
        # 1. Config
        pop_size = self.config.get("pop_size", 50)  # Số lượng nguồn thức ăn (SN)
        limit = self.config.get("limit", 100)       # Số lần thử tối đa trước khi bỏ nguồn thức ăn
        max_iters = self.config.get("max_iters", 1000)

        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()

        # 2. Initialization
        # Quần thể là các nguồn thức ăn
        population = np.random.uniform(low, high, (pop_size, dim))
        fitness = np.array([self.evaluate(ind) for ind in population])
        trial_counters = np.zeros(pop_size) # Đếm số lần không cải thiện

        if any(f is None for f in fitness): return self._build_result()

        # Tìm best ban đầu
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        # 3. Main Loop
        for iteration in range(max_iters):
            if self.evaluations >= self.max_evals:
                break

            # --- Giai đoạn Ong thợ (Employed Bees) ---
            for i in range(pop_size):
                if self.evaluations >= self.max_evals: break
                
                # Chọn ngẫu nhiên một đối tác k khác i
                k = i
                while k == i:
                    k = np.random.randint(0, pop_size)
                
                # Tạo giải pháp mới: v_i = x_i + phi * (x_i - x_k)
                phi = np.random.uniform(-1, 1, dim)
                new_sol = population[i] + phi * (population[i] - population[k])
                new_sol = np.clip(new_sol, low, high)

                f_new = self.evaluate(new_sol)
                if f_new is None: break

                # Greedy Selection
                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # --- Giai đoạn Ong quan sát (Onlooker Bees) ---
            # Tính xác suất chọn dựa trên fitness (fitness càng nhỏ càng tốt -> nghịch đảo)
            # Để tránh chia cho 0 hoặc số âm (nếu hàm mục tiêu có âm), ta dùng exp hoặc rank.
            # Ở đây dùng công thức đơn giản cho bài toán minimize: P_i = (1/f_i) / sum(1/f)
            # Hoặc fitness transform: fit = 1/(1+f) nếu f>=0 else 1+abs(f)
            
            # Xử lý an toàn:
            w_fitness = np.zeros_like(fitness)
            for idx, val in enumerate(fitness):
                if val >= 0:
                    w_fitness[idx] = 1.0 / (1.0 + val)
                else:
                    w_fitness[idx] = 1.0 + np.abs(val)
            
            prob = w_fitness / np.sum(w_fitness)
            
            # Onlooker chọn nguồn thức ăn dựa trên Prob
            # Số lượng onlooker thường bằng pop_size
            for _ in range(pop_size):
                if self.evaluations >= self.max_evals: break

                # Roulette wheel selection
                i = self._roulette_wheel_selection(prob)
                
                # Logic giống Employed Bee
                k = i
                while k == i:
                    k = np.random.randint(0, pop_size)
                
                phi = np.random.uniform(-1, 1, dim)
                new_sol = population[i] + phi * (population[i] - population[k])
                new_sol = np.clip(new_sol, low, high)

                f_new = self.evaluate(new_sol)
                if f_new is None: break

                if f_new < fitness[i]:
                    population[i] = new_sol
                    fitness[i] = f_new
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # --- Giai đoạn Ong trinh sát (Scout Bees) ---
            # Tìm nguồn thức ăn đã vượt quá giới hạn thử 'limit'
            max_trials_idx = np.argmax(trial_counters)
            if trial_counters[max_trials_idx] > limit:
                # Reset bằng ngẫu nhiên
                population[max_trials_idx] = np.random.uniform(low, high, dim)
                f_reset = self.evaluate(population[max_trials_idx])
                if f_reset is not None:
                    fitness[max_trials_idx] = f_reset
                    trial_counters[max_trials_idx] = 0

            # Update Global Best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                self.best_solution = population[current_best_idx].copy()
            
            self.history.append(self.best_fitness)

        return self._build_result()

    def _roulette_wheel_selection(self, prob):
        r = np.random.rand()
        cumsum = np.cumsum(prob)
        return np.searchsorted(cumsum, r)