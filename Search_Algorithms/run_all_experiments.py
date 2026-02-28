"""
Run all experiments for continuous and discrete problems.
Each problem runs 4 times with available algorithms.
"""

from experiment.run_experiment import run_experiment
from experiment.logger import save_summary_txt
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_radar_chart(problem_name, results_dict, plots_dir):
    """
    Tạo Radar Chart so sánh các thuật toán
    """
    algorithms = list(results_dict.keys())
    
    # 6 Tiêu chí dựa theo Phần III của PDF
    metrics_keys = ['mean_score', 'best_score', 'std_score', 'mean_time', 'mean_effort', 'success_rate']
    
    # Tên nhãn hiển thị trên biểu đồ
    formatted_metrics = [
        'Avg Quality\n(mean_score)', 
        'Peak Quality\n(best_score)', 
        'Stability\n(std_score)', 
        'Speed\n(mean_time)', 
        'Comp. Cost\n(mean_effort)', 
        'Reliability\n(success_rate)'
    ]
    
    # Gom dữ liệu
    raw_data = {key: [] for key in metrics_keys}
    for algo in algorithms:
        summary = results_dict[algo]
        raw_data['mean_score'].append(summary.get('mean_score', 0) or 0)
        raw_data['best_score'].append(summary.get('best_score', 0) or 0)
        raw_data['std_score'].append(summary.get('std_score', 0) or 0)
        raw_data['mean_time'].append(summary.get('mean_time', 0) or 0)
        raw_data['mean_effort'].append(summary.get('mean_effort', 0) or 0)
        raw_data['success_rate'].append(summary.get('success_rate', 0) or 0)

    # Chuẩn hóa dữ liệu về [0.1, 1.0] để vẽ
    normalized_data = {}
    for key in metrics_keys:
        arr = np.array(raw_data[key])
        if np.max(arr) == np.min(arr):
            # Nếu tất cả các thuật toán bằng điểm nhau ở tiêu chí này
            normalized_data[key] = np.ones_like(arr) if np.max(arr) > 0 else np.full_like(arr, 0.1)
        else:
            if key == 'success_rate':
                # Success Rate: Điểm cao là TỐT
                norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            else:
                # Các chỉ số còn lại (Score, Time, Effort, Std): Điểm thấp là TỐT
                # Ta nghịch đảo lại để trên biểu đồ, điểm nằm ở viền ngoài cùng luôn là thuật toán đỉnh nhất
                norm = (np.max(arr) - arr) / (np.max(arr) - np.min(arr))
            
            # Ép vào khoảng [0.1, 1.0] để đồ thị không bị chọc thủng về tâm (số 0)
            normalized_data[key] = 0.1 + (norm * 0.9)

    # Chuẩn bị góc vẽ cho Lục giác (6 góc)
    N = len(metrics_keys)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    # Vẽ các đường đa giác
    for i, algo in enumerate(algorithms):
        values = [normalized_data[key][i] for key in metrics_keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo)
        ax.fill(angles, values, alpha=0.15)

    # Căn chỉnh hiển thị
    plt.xticks(angles[:-1], formatted_metrics, size=11, weight='bold')
    ax.tick_params(axis='x', pad=20) # Đẩy chữ ra xa viền cho dễ đọc
    
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
    
    plt.title(f"{problem_name.upper()} - Performance Comparison", size=16, weight='bold', y=1.15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    # Lưu biểu đồ
    import os
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{problem_name}_radar.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    [+] Saved HEXAGON radar chart: {plot_path}")

def main():
    # Define all experiment configurations
    # Format: (config_path, output_filename)
    experiments = [
        # Continuous problems
        ("config/sphere_experiment.yaml", "sphere_results.txt"),
        ("config/ackley_experiment.yaml", "ackley_results.txt"),
        ("config/rastrigin_experiment.yaml", "rastrigin_results.txt"),
        ("config/rosenbrock_experiment.yaml", "rosenbrock_results.txt"),
        ("config/griewank_experiment.yaml", "griewank_results.txt"),
        # Discrete problems
        ("config/grid_pathfinding.yaml", "grid_pathfinding_results.txt"),
        ("config/n_queens.yaml", "n_queens_results.txt"),
    ]
    
    txt_dir = "data/summary_results"
    plots_dir = "data/plots"
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    
    all_results = {}
    
    for config_path, output_file in experiments:
        print(f"\n{'='*70}")
        print(f"Running: {config_path}")
        print(f"{'='*70}")
        
        try:
            results = run_experiment(config_path)
            all_results[config_path] = results
            problem_name = config_path.split('/')[-1].replace('_experiment.yaml', '').replace('.yaml', '')
            
            # Save txt
            output_path = os.path.join(txt_dir, output_file)
            for algo, summary in results.items():
                save_summary_txt(algo, summary, output_path)
            
            # Print console
            for algo, summary in results.items():
                print(f"\n  Algorithm: {algo}")
                print(f"    Runs: {summary['runs']}")
                print(f"    Best Score: {summary['best_score']:.6f}" if summary['best_score'] is not None else "    Best Score: N/A")
                print(f"    Mean Score: {summary['mean_score']:.6f}" if summary['mean_score'] is not None else "    Mean Score: N/A")
                print(f"    Std Score: {summary['std_score']:.6f}")
                print(f"    Mean Time: {summary['mean_time']:.6f}s")
            
            print(f"\n  [+] Saved txt: {output_path}")
            
            # Vẽ đồ thị & lưu vào folder plots
            plot_radar_chart(problem_name, results, plots_dir)
            
        except Exception as e:
            print(f"ERROR running {config_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"\nAll results saved in: {txt_dir}/ and {plots_dir}/")
    
    # Print final summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Problem':<25} {'Algorithm':<20} {'Best':<12} {'Mean':<12} {'Time(s)':<10} {'Success':<8}")
    print("-" * 90)
    
    for config_path, results in all_results.items():
        problem_name = config_path.replace("config/", "").replace("_experiment.yaml", "").replace(".yaml", "")
        for algo, summary in results.items():
            best = f"{summary['best_score']:.4f}" if summary['best_score'] is not None else "N/A"
            mean = f"{summary['mean_score']:.4f}" if summary['mean_score'] is not None else "N/A"
            time = f"{summary['mean_time']:.4f}"
            success = f"{summary['success_rate']:.0%}"
            print(f"{problem_name:<25} {algo:<20} {best:<12} {mean:<12} {time:<10} {success:<8}")

if __name__ == "__main__":
    main()