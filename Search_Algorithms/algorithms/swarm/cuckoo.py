import numpy as np
import math
from algorithms.base.optimizer_base import Optimizer

class CuckooSearch(Optimizer):
    def run(self):
        # 1. Cấu hình tham số
        n_nests = self.config.get("n_nests", 25)      # Số lượng tổ chim (population size)
        pa = self.config.get("pa", 0.25)              # Tỉ lệ phát hiện (discovery rate)
        beta = self.config.get("beta", 1.5)           # Tham số cho Lévy flight
        max_iters = self.config.get("max_iters", 1000)

        # 2. Khởi tạo quần thể (nests)
        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()
        
        # Tạo quần thể ban đầu
        nests = np.random.uniform(low, high, (n_nests, dim))
        fitness = np.array([self.evaluate(x) for x in nests])
        
        # Xử lý trường hợp evaluate trả về None (nếu quá số lần đánh giá ngay từ đầu)
        if None in fitness:
            self.best_solution = nests[0]
            self.best_fitness = float('inf')
            return self._build_result()

        # Tìm best ban đầu
        best_idx = np.argmin(fitness)
        self.best_solution = nests[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history.append(self.best_fitness)

        # --- Chuẩn bị các hằng số cho Lévy Flight (Dùng math thay vì scipy) ---
        # Công thức Sigma:
        # sigma = (gamma(1+beta) * sin(pi*beta/2) / (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta)
        
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        sigma_v = 1  # Theo chuẩn thuật toán

        # 3. Vòng lặp chính
        for t in range(max_iters):
            if self.evaluations >= self.max_evals:
                break

            # --- BƯỚC 1: Tạo cuckoo mới bằng Lévy Flight ---
            # Lấy ngẫu nhiên các tổ để biến đổi
            new_nests = nests.copy()
            
            # Tạo bước nhảy Lévy
            u = np.random.normal(0, sigma_u, (n_nests, dim))
            v = np.random.normal(0, sigma_v, (n_nests, dim))
            step = u / (np.abs(v) ** (1 / beta))
            
            # Cập nhật vị trí: X_new = X + step_size * step * (X - X_best)
            # Lưu ý: Cuckoo Search gốc thường dùng step size alpha = 0.01 * scale
            step_size = 0.01 * step * (new_nests - self.best_solution)
            
            # Tạo ứng viên mới
            new_nests = new_nests + step_size * np.random.randn(n_nests, dim)
            
            # Clip trong biên
            new_nests = np.clip(new_nests, low, high)

            # Đánh giá và chọn lọc (Greedy Selection)
            for i in range(n_nests):
                f_new = self.evaluate(new_nests[i])
                if f_new is None: break
                
                # Nếu tốt hơn thì thay thế tổ cũ
                if f_new < fitness[i]:
                    fitness[i] = f_new
                    nests[i] = new_nests[i]

            # --- BƯỚC 2: Loại bỏ một phần các tổ xấu (Alien eggs discovery) ---
            # Tạo một bộ tổ mới ngẫu nhiên dựa trên xác suất Pa
            # K = tập các tổ bị phát hiện (xấu nhất)
            
            # Sắp xếp để biết tổ nào tốt/xấu (để giữ lại best)
            sorted_idx = np.argsort(fitness)
            
            # Giữ lại tổ tốt nhất (Elitism), các tổ còn lại có xác suất bị thay thế
            # Tạo mask ngẫu nhiên: True = bị phát hiện/thay thế
            mask = np.random.rand(n_nests, dim) < pa
            
            # Tuy nhiên, ta không thay đổi tổ tốt nhất hiện tại
            mask[sorted_idx[0]] = False 
            
            # Tạo bước nhảy ngẫu nhiên (Biased Random Walk)
            # X_new = X + r * (X_rand1 - X_rand2)
            rand_idx1 = np.random.randint(0, n_nests, n_nests)
            rand_idx2 = np.random.randint(0, n_nests, n_nests)
            
            step_rand = np.random.rand() * (nests[rand_idx1] - nests[rand_idx2])
            new_nests_abandon = nests + step_rand * mask # Chỉ cộng vào những chố mask=True
            
            new_nests_abandon = np.clip(new_nests_abandon, low, high)

            # Đánh giá lại các tổ vừa bị thay đổi
            for i in range(n_nests):
                # Chỉ đánh giá nếu tổ đó thực sự thay đổi (dựa vào mask check qua loa hoặc check diff)
                # Để đơn giản code: đánh giá những thằng bị mask (có ít nhất 1 chiều bị thay đổi)
                if np.any(mask[i]):
                    f_new = self.evaluate(new_nests_abandon[i])
                    if f_new is None: break
                    
                    if f_new < fitness[i]:
                        fitness[i] = f_new
                        nests[i] = new_nests_abandon[i]

            # --- BƯỚC 3: Cập nhật Best Global ---
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = nests[min_fitness_idx].copy()
            
            self.history.append(self.best_fitness)

        return self._build_result()