"""
Run all experiments for continuous and discrete problems.
Each problem runs 4 times with available algorithms.
"""

from experiment.run_experiment import run_experiment
from experiment.logger import save_summary_txt
import os
from visualization.plot_comparison import plot_continuous_radar

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
                print(f"    Mean Effort : {summary.get('mean_effort', 0):.2f}") 
                print(f"    Success Rate: {summary.get('success_rate', 0):.2%}")
            
            print(f"\n  [+] Saved txt: {output_path}")
            
            # Vẽ đồ thị & lưu vào folder plots
            plot_path = os.path.join(plots_dir, f"{problem_name}_radar.png")
            plot_continuous_radar(problem_name, results, plot_path)
            print(f"    [+] Saved HEXAGON radar chart: {plot_path}")
            
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
    print(f"{'Problem':<25} {'Algorithm':<20} {'Best':<12} {'Mean':<12} {'Time(s)':<10} {'Effort':<10} {'Success':<8}")
    print("-" * 95)
    
    for config_path, results in all_results.items():
        problem_name = config_path.replace("config/", "").replace("_experiment.yaml", "").replace(".yaml", "")
        for algo, summary in results.items():
            best = f"{summary['best_score']:.4f}" if summary['best_score'] is not None else "N/A"
            mean = f"{summary['mean_score']:.4f}" if summary['mean_score'] is not None else "N/A"
            time = f"{summary['mean_time']:.4f}"
            effort = f"{summary.get('mean_effort', 0):.2f}"
            success = f"{summary['success_rate']:.0%}"
            print(f"{problem_name:<25} {algo:<20} {best:<12} {mean:<12} {time:<10} {effort:<10} {success:<8}")

if __name__ == "__main__":
    main()