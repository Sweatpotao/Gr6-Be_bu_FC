import matplotlib.pyplot as plt
import numpy as np
import os

def normalize_metrics(results_dict):
    # Chuẩn hóa các tiêu chí (càng nhỏ càng tốt) về khoảng [0, 1].
    # 1.0 là tốt nhất (ngoài cùng đồ thị), 0.0 là tệ nhất (tâm đồ thị).

    metrics = ['mean_score', 'best_score', 'std_score', 'mean_time', 'mean_effort', 'success_rate']
    labels = ['Mean Score', 'Best Score', 'Stability\n(Std score)', 'Speed\n(Mean time)',
              'Comp. Cost\n(Mean effort)', 'Reliability\n(Success rate)']
    
    # Lấy data
    safe_dict = {}
    for algo, stats in results_dict.items():
        safe_dict[algo] = {k: float(stats.get(k, 0.0) or 0.0) for k in metrics}

    # Tìm Min và Max cho từng tiêu chí
    min_v = {k: min(safe_dict[algo][k] for algo in safe_dict) for k in metrics}
    max_v = {k: max(safe_dict[algo][k] for algo in safe_dict) for k in metrics}
    
    # Chuẩn hóa
    normalized = {}
    for algo in safe_dict:
        norm_scores = []
        for k in metrics:
            val = safe_dict[algo][k]
            denom = max_v[k] - min_v[k]
            
            if denom == 0:
                # Nếu tất cả các thuật toán bằng điểm nhau ở tiêu chí này
                score = 1.0 if max_v[k] > 0 else 0.1
            else:
                if k == 'success_rate':
                    # Ngoại lệ 1: Success Rate thì CÀNG LỚN CÀNG TỐT
                    norm = (val - min_v[k]) / denom
                else:
                    # Các chỉ số còn lại: CÀNG NHỎ CÀNG TỐT (đảo ngược)
                    norm = (max_v[k] - val) / denom
                
                # Ép dải giá trị vào [0.1, 1.0] để không bị chìm lấp ở tâm (0)
                score = 0.1 + (norm * 0.9)
                
            norm_scores.append(score)
            
        normalized[algo] = norm_scores
        
    return normalized, labels

def plot_continuous_radar(problem_name, results_dict, save_path):
    # Vẽ và lưu Radar Chart
    data_matrix, metrics = normalize_metrics(results_dict)
    algorithms = list(data_matrix.keys())
    
    if not algorithms:
        return

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Đóng vòng

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(0)
    ax.set_theta_direction(-1)

    # Setup trục tọa độ - Căn chỉnh lại khoảng cách chữ
    plt.xticks(angles[:-1], metrics, color='black', size=11, weight='bold')
    ax.tick_params(axis='x', pad=20) 
    
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)

    # Vẽ từng thuật toán
    colors = plt.cm.tab10.colors
    for idx, algo in enumerate(algorithms):
        values = data_matrix[algo]
        values += values[:1] # Đóng vòng
        
        c = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=algo, color=c)
        ax.fill(angles, values, alpha=0.15, color=c)

    plt.title(f"{problem_name.upper()} - Performance Comparison", size=16, weight='bold', y=1.15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()