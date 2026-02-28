import matplotlib.pyplot as plt
import numpy as np
import os

def normalize_metrics(results_dict):
    # Chuẩn hóa các tiêu chí (càng nhỏ càng tốt) về khoảng [0, 1].
    # 1.0 là tốt nhất (ngoài cùng đồ thị), 0.0 là tệ nhất (tâm đồ thị).

    metrics = ['mean_score', 'best_score', 'std_score', 'mean_time', 'mean_effort']
    labels = ['Mean Score\n(Quality)', 'Best Score\n(Peak)', 'Std Score\n(Stability)', 'Mean Time\n(Speed)', 'Mean Effort\n(Evals)']
    
    # Tìm Min và Max cho từng tiêu chí
    min_v = {k: float('inf') for k in metrics}
    max_v = {k: float('-inf') for k in metrics}
    
    for algo, stats in results_dict.items():
        for k in metrics:
            val = stats.get(k, 0.0) if stats.get(k) is not None else 0.0
            if val < min_v[k]: min_v[k] = val
            if val > max_v[k]: max_v[k] = val

    # Chuẩn hóa
    normalized = {}
    for algo, stats in results_dict.items():
        norm_scores = []
        for k in metrics:
            val = stats.get(k, 0.0) if stats.get(k) is not None else 0.0
            denom = max_v[k] - min_v[k]
            
            # Đảo ngược: (Max - Val) / (Max - Min)
            score = 1.0 if denom == 0 else (max_v[k] - val) / denom
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

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Setup trục tọa độ
    plt.xticks(angles[:-1], metrics, color='black', size=10, weight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "Best (1.0)"], color="grey", size=8)

    # Vẽ từng thuật toán
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, algo in enumerate(algorithms):
        values = data_matrix[algo]
        values += values[:1] # Đóng vòng
        
        c = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo, color=c)
        ax.fill(angles, values, alpha=0.15, color=c)

    plt.title(f"Algorithm Comparison: {problem_name.upper()}", size=14, weight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()