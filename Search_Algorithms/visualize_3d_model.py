import numpy as np
import matplotlib.pyplot as plt

from experiment.registry import PROBLEM_REGISTRY, ALGORITHM_REGISTRY

# ==========================================
# 1. CẤU HÌNH BÀI TOÁN & THUẬT TOÁN
# ==========================================
# Cấu hình chuẩn cho không gian 2D để vẽ đồ thị
PROBLEM_CONFIGS = {
    "sphere": {"dim": 2, "bounds": [[-5.12, 5.12], [-5.12, 5.12]]},
    "ackley": {"dim": 2, "bounds": [[-32.768, 32.768], [-32.768, 32.768]], "a": 20.0, "b": 0.2, "c": 6.283185307},
    "rastrigin": {"dim": 2, "bounds": [[-5.12, 5.12], [-5.12, 5.12]]},
    "rosenbrock": {"dim": 2, "bounds": [[-2.048, 2.048], [-2.048, 2.048]]},
    "griewank": {"dim": 2, "bounds": [[-600, 600], [-600, 600]]}
}

# Danh sách 10 thuật toán liên tục
CONTINUOUS_ALGOS = [
    "HillClimbing", "SimulatedAnnealing", 
    "GA", "DE", 
    "ABC", "CuckooSearch", "FireflyAlgorithm", "PSO", 
    "TLBO"
]

# Cấu hình chung cho thuật toán (Chạy nhanh, số lượng quần thể nhỏ để dễ nhìn đường đi)
ALGO_CONFIG = {
    "pop_size": 20, "n_particles": 20, "n_nests": 20, "archive_size": 20,
    "max_iters": 50, "max_evaluations": 1000, "timeout": 30,
    "step_size": 0.5, "initial_temp": 100, "cooling_rate": 0.95,
    "mutation_rate": 0.1, "crossover_rate": 0.9, "elite_size": 2,
    "F": 0.8, "Cr": 0.9,
    "limit": 20, "sample_size": 10, "pa": 0.25, 
    "alpha": 0.2, "beta0": 1.0, "gamma": 1.0, "w": 0.7, "c1": 1.5, "c2": 1.5
}

# ==========================================
# 2. LỚP BỌC ĐIỆP VIÊN (TRACER WRAPPER) ĐÃ ĐƯỢC NÂNG CẤP
# ==========================================
class TracerWrapper:
    """Bọc bên ngoài Problem gốc để lén ghi lại các tọa độ được đánh giá"""
    def __init__(self, real_problem):
        self.real_problem = real_problem
        self.explored_points = []

    def get_dimension(self):
        return self.real_problem.get_dimension()

    def get_bounds(self):
        bounds = self.real_problem.get_bounds()
        
        flat_bounds = np.array(bounds).flatten()
        
        low_scalar = float(np.min(flat_bounds))
        high_scalar = float(np.max(flat_bounds))
        
        return low_scalar, high_scalar

    def initial_solution(self):
        return self.real_problem.initial_solution()

    def evaluate(self, x):
        val = self.real_problem.evaluate(x)
        if val is not None:
            # Ghi lại tọa độ x1, x2 và giá trị fitness
            self.explored_points.append((x[0], x[1], val))
        return val

# ==========================================
# 3. TRÌNH ĐIỀU KHIỂN CHÍNH
# ==========================================
def main():
    # --- MENU CHỌN BÀI TOÁN ---
    print("=== CHỌN BÀI TOÁN ===")
    prob_names = list(PROBLEM_CONFIGS.keys())
    for i, name in enumerate(prob_names):
        print(f"{i+1}. {name.capitalize()}")
    
    prob_choice = int(input(f"Nhập số (1-{len(prob_names)}): ")) - 1
    selected_prob_name = prob_names[prob_choice % len(prob_names)]
    
    # --- MENU CHỌN THUẬT TOÁN ---
    print("\n=== CHỌN THUẬT TOÁN ===")
    for i, algo in enumerate(CONTINUOUS_ALGOS):
        print(f"{i+1}. {algo}")
    print(f"{len(CONTINUOUS_ALGOS) + 1}. CHẠY TẤT CẢ")
    
    algo_choice = int(input(f"Nhập số (1-{len(CONTINUOUS_ALGOS) + 1}): ")) - 1
    
    # Quyết định danh sách thuật toán sẽ chạy
    if algo_choice == len(CONTINUOUS_ALGOS):
        selected_algos = CONTINUOUS_ALGOS # Chạy tất cả
    else:
        selected_algos = [CONTINUOUS_ALGOS[algo_choice]] # Chỉ chạy 1 cái đã chọn
    
    # Khởi tạo bài toán gốc 2D
    ProblemClass = PROBLEM_REGISTRY[selected_prob_name]
    real_problem = ProblemClass(**PROBLEM_CONFIGS[selected_prob_name])
    bounds = PROBLEM_CONFIGS[selected_prob_name]["bounds"]
    bound_val = bounds[0][1] # Lấy giá trị biên dương (VD: 5.12)
    
    print(f"\n🚀 Khởi động 3D trên hàm: {selected_prob_name.upper()}")
    
    # Chạy các thuật toán (1 cái hoặc nhiều cái tùy lựa chọn)
    algo_paths = {}
    
    for algo_name in selected_algos:
        if algo_name not in ALGORITHM_REGISTRY:
            print(f"  [!] Bỏ qua {algo_name} vì không có trong REGISTRY.")
            continue
            
        print(f"  [-] Đang chạy {algo_name}...")
        AlgoClass = ALGORITHM_REGISTRY[algo_name]
        
        # Bọc điệp viên vào bài toán
        tracer = TracerWrapper(real_problem)
        optimizer = AlgoClass(tracer, ALGO_CONFIG)
        optimizer.run()
        
        # Trích xuất "Quỹ đạo tốt nhất" (Chỉ lấy điểm khi fitness cải thiện)
        path_x, path_y, path_z = [], [], []
        current_best = float('inf')
        for px, py, pz in tracer.explored_points:
            if pz < current_best:
                current_best = pz
                path_x.append(px)
                path_y.append(py)
                path_z.append(pz)
                
        algo_paths[algo_name] = (path_x, path_y, path_z)

# ==========================================
# 4. VẼ ĐỒ THỊ 3D VÀ BẬT TƯƠNG TÁC
# ==========================================
    print("\nĐang tạo đồ thị 3D...")
    
    # Tạo lưới mặt cong
    x_vals = np.linspace(-bound_val, bound_val, 100)
    y_vals = np.linspace(-bound_val, bound_val, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Lấy lưới Z bằng cách dò tay qua hàm gốc
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = real_problem.evaluate(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ mặt cong với độ trong suốt cao để không che lấp đường đi
    ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.4)
    
    # Vẽ quỹ đạo của TỪNG thuật toán
    colors = plt.cm.tab10.colors # Bộ 10 màu phân biệt của matplotlib
    for i, (algo_name, (px, py, pz)) in enumerate(algo_paths.items()):
        if len(px) == 0: continue
        c = colors[i % len(colors)]
        
        # Nâng đường đi lên để nổi rõ trên mặt cong
        z_offset = np.max(Z) * 0.05
        ax.plot(px, py, np.array(pz) + z_offset, color=c, linewidth=2, marker='.', markersize=6, label=algo_name)
        # Đánh dấu điểm kết thúc
        if i == 0:
            ax.scatter(px[-1], py[-1], pz[-1] + z_offset,
                    color=c, marker='*', s=150,
                    edgecolors='black',
                    label='Final Point (*)')
        else:
            ax.scatter(px[-1], py[-1], pz[-1] + z_offset,
                    color=c, marker='*', s=150,
                    edgecolors='black')

    # Đánh dấu mục tiêu tối thượng (0, 0) - Thường đáy của hàm benchmark nằm ở (0,0) hoặc (1,1)
    if selected_prob_name == "rosenbrock":
        ax.scatter(1, 1, 0.2, color='red', marker='.', s=200, label='Global Optimum (1,1)')
    else:
        ax.scatter(0, 0, 0.2, color='red', marker='.', s=200, label='Global Optimum (0,0)')

    ax.set_title(f"Algorithm Racing on {selected_prob_name.upper()} Landscape", fontsize=16, fontweight='bold')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Fitness')
    
    # Tinh chỉnh chú thích (Legend)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title="Algorithms")
    
    print("Mở cửa sổ thành công! Có thể tương tác với đồ thị.")
    plt.show()

if __name__ == "__main__":
    main()