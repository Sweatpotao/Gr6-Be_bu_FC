import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Danh sách các thuật toán liên tục của bạn
ALGORITHMS = ['Hill Climbing', 'SA', 'ABC', 'Cuckoo Search', 'Firefly', 'PSO', 'GA', 'DE', 'TLBO']
# Số lần chạy độc lập
NUM_RUNS = 30
# Số vòng lặp (giả định để vẽ biểu đồ)
MAX_ITER = 1000
# Chi phí mặc định
EFFORT = 10000

def calculate_conv(success_rate, mean_fitness, effort):
    """
    Tính chỉ số Conv theo công thức yêu cầu.
    success_rate: Dạng thập phân (VD: 100% -> 1.0)
    """
    return (100 * success_rate) / ((mean_fitness + 0.1) * np.sqrt(effort))

def run_real_algorithm(algo_name, problem_name):
    """
    TODO: THAY THẾ HÀM NÀY BẰNG CODE CHẠY THUẬT TOÁN THỰC TẾ CỦA BẠN.
    Hàm này cần trả về:
    - history_runs: mảng 2D kích thước (NUM_RUNS, MAX_ITER) chứa lịch sử fitness
    - success_rate: tỷ lệ thành công (0.0 đến 1.0)
    """
    # --- MOCK DATA ---
    iterations = np.arange(MAX_ITER)
    # Tạo độ dốc giảm dần tùy theo thuật toán
    decay = np.random.uniform(0.005, 0.02)
    base_curve = 1000 * np.exp(-decay * iterations) + np.random.uniform(0, 50)
    
    history_runs = []
    for _ in range(NUM_RUNS):
        # Thêm nhiễu ngẫu nhiên cho từng lần chạy để tạo vùng bóng mờ (std)
        noise = np.random.normal(0, base_curve * 0.1, MAX_ITER)
        run_curve = np.clip(base_curve + noise, 0, None)
        # Giả lập hội tụ dần về cuối
        run_curve = np.minimum.accumulate(run_curve) 
        history_runs.append(run_curve)
        
    history_runs = np.array(history_runs)
    
    # Giả lập kết quả Mean và Success
    mean_fitness = np.mean(history_runs[:, -1])
    success_rate = 1.0 if mean_fitness < 10 else np.random.choice([0.0, 0.2, 0.5])
    
    return history_runs, success_rate, mean_fitness

def draw_convergence(problem_name):
    import matplotlib.ticker as ticker # Thêm thư viện này ở đầu file

def draw_convergence(problem_name):
    plt.figure(figsize=(10, 6))
    
    print(f"\n--- ĐANG CHẠY BÀI TOÁN: {problem_name.upper()} ---")
    print(f"{'Algorithm':<18} | {'Mean':<10} | {'Success':<8} | {'Conv Score'}")
    print("-" * 55)
    
    # --- CẢI TIẾN 1: Bỏ qua N vòng lặp đầu để đồ thị không bị "rơi thẳng đứng" ---
    start_iter = 10
    x_axis = np.arange(start_iter, MAX_ITER)
    
    for algo in ALGORITHMS:
        # 1. Chạy thuật toán và lấy lịch sử
        history_runs, success_rate, mean_fitness = run_real_algorithm(algo, problem_name)
        
        # 2. Tính toán đường trung bình
        mean_curve = np.mean(history_runs, axis=0)
        
    # --- CẢI TIẾN 2: Dùng Percentile thay cho Std để vùng bóng mờ không bị âm ---
        low_bound = np.percentile(history_runs, 25, axis=0)
        high_bound = np.percentile(history_runs, 75, axis=0)
        
        # 3. Tính toán chỉ số Conv
        conv_score = calculate_conv(success_rate, mean_fitness, EFFORT)
        
        # In kết quả ra console
        success_str = f"{success_rate*100:.0f}%"
        print(f"{algo:<18} | {mean_fitness:<10.4f} | {success_str:<8} | {conv_score:.4f}")
        
        # 4. Vẽ đồ thị TỪ start_iter trở đi
        line, = plt.plot(x_axis, mean_curve[start_iter:], label=algo, linewidth=1.5)
        plt.fill_between(x_axis, 
                         low_bound[start_iter:], 
                         high_bound[start_iter:], 
                         color=line.get_color(), 
                         alpha=0.15, edgecolor='none') 

    # Thiết lập giao diện biểu đồ
    plt.title(f'Convergence Curves - {problem_name.capitalize()} Function', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness (Objective Value)', fontsize=12, fontweight='bold')
    
    # --- CẢI TIẾN 3: Phân lớp trục Y chi tiết hơn ---
    ax = plt.gca()
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15))
    
    # Bật lưới cho cả vạch chính (major) và vạch phụ (minor)
    ax.grid(True, which="major", linestyle='-', alpha=0.6)
    ax.grid(True, which="minor", linestyle=':', alpha=0.3)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    # Hiển thị biểu đồ
    plt.show()

if __name__ == "__main__":
    valid_problems = ['sphere', 'ackley', 'rastrigin', 'rosenbrock', 'griewank']
    while 1:
        print("Các bài toán hỗ trợ: sphere, ackley, rastrigin, rosenbrock, griewank")
        selected_problem = int(input("Nhập bài toán muốn chạy: "))
        if selected_problem > 0 and selected_problem < 6:
            draw_convergence(valid_problems[selected_problem - 1])
        else:
            print("Exit!")
            break