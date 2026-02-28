"""
Run all experiments for continuous and discrete problems.
Each problem runs 4 times with available algorithms.
"""

from experiment.run_experiment import run_experiment
from experiment.logger import save_summary_txt
from visualization.continuous_comparison import plot_radar_chart
from visualization.discrete_comparison import create_spider_chart
import os

# Discrete problems list
DISCRETE_PROBLEMS = {'grid_pathfinding', 'n_queens', 'tsp', 'knapsack', 'graph_coloring'}

def convert_results_to_spider_data(results):
    """
    Convert experiment results to spider chart format for discrete problems.
    
    Returns dict: {algorithm_name: [quality, speed, efficiency, reliability, convergence]}
    """
    spider_data = {}
    
    if not results:
        return spider_data
    
    # Find max values for normalization
    max_time = max(r['mean_time'] for r in results.values() if r['mean_time'] > 0) or 1.0
    max_effort = max(r['mean_effort'] for r in results.values() if r['mean_effort'] > 0) or 1.0
    
    # For solution quality, we need to normalize scores
    scores = [r['best_score'] for r in results.values() if r['best_score'] is not None]
    if scores:
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score > min_score else 1
    else:
        min_score, max_score, score_range = 0, 1, 1
    
    for algo_name, summary in results.items():
        # Solution Quality: normalized score (higher is better)
        if summary['best_score'] is not None:
            quality = (summary['best_score'] - min_score) / score_range
        else:
            quality = summary['success_rate']
        
        # Speed: inverse of time (faster is better)
        speed = 1 - (summary['mean_time'] / max_time) if max_time > 0 else 0
        
        # Efficiency: inverse of effort (fewer evaluations is better)
        efficiency = 1 - (summary['mean_effort'] / max_effort) if max_effort > 0 else 0
        
        # Reliability: success rate
        reliability = summary['success_rate']
        
        # Convergence: same as reliability for discrete
        convergence = summary['success_rate']
        
        spider_data[algo_name] = [
            round(max(0, min(1, quality)), 4),
            round(max(0, min(1, speed)), 4),
            round(max(0, min(1, efficiency)), 4),
            round(reliability, 4),
            round(convergence, 4)
        ]
    
    return spider_data

def is_discrete_problem(problem_name):
    """Check if a problem is discrete based on its name."""
    return problem_name in DISCRETE_PROBLEMS

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
        # Discrete problems - Classical + Metaheuristic
        ("config/grid_pathfinding.yaml", "grid_pathfinding_results.txt"),
        ("config/n_queens.yaml", "n_queens_results.txt"),
        ("config/tsp.yaml", "tsp_results.txt"),
        ("config/knapsack.yaml", "knapsack_results.txt"),
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
            if is_discrete_problem(problem_name):
                # Use discrete spider chart
                spider_data = convert_results_to_spider_data(results)
                chart_path = os.path.join(plots_dir, f"{problem_name}_spider.png")
                create_spider_chart(spider_data, title=f"{problem_name.upper()} - Algorithm Comparison", save_path=chart_path)
                print(f"  [+] Saved discrete spider chart: {chart_path}")
            else:
                # Use continuous radar chart
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