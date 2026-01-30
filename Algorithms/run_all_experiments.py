"""
Run all experiments for continuous and discrete problems.
Each problem runs 4 times with available algorithms.
"""

from experiment.run_experiment import run_experiment
from experiment.logger import save_summary_txt
import os

def main():
    # Define all experiment configurations
    # Format: (config_path, output_filename)
    experiments = [
        # Continuous problems
        ("config/ackley_experiment.yaml", "ackley_results.txt"),
        ("config/rastrigin_experiment.yaml", "rastrigin_results.txt"),
        ("config/rosenbrock_experiment.yaml", "rosenbrock_results.txt"),
        ("config/griewank_experiment.yaml", "griewank_results.txt"),
        # Discrete problems
        ("config/grid_pathfinding.yaml", "grid_pathfinding_results.txt"),
        ("config/n_queens.yaml", "n_queens_results.txt"),
    ]
    
    output_dir = "data/summary_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("Each problem runs 4 times with available algorithms")
    print("=" * 70)
    print()
    
    all_results = {}
    
    for config_path, output_file in experiments:
        print(f"\n{'='*70}")
        print(f"Running: {config_path}")
        print(f"{'='*70}")
        
        try:
            results = run_experiment(config_path)
            all_results[config_path] = results
            
            # Save results to file
            output_path = os.path.join(output_dir, output_file)
            for algo, summary in results.items():
                save_summary_txt(algo, summary, output_path)
            
            # Print summary
            print(f"\nResults Summary for {config_path}:")
            print("-" * 50)
            for algo, summary in results.items():
                print(f"\n  Algorithm: {algo}")
                print(f"    Runs: {summary['runs']}")
                print(f"    Best Score: {summary['best_score']:.6f}" if summary['best_score'] is not None else "    Best Score: N/A")
                print(f"    Mean Score: {summary['mean_score']:.6f}" if summary['mean_score'] is not None else "    Mean Score: N/A")
                print(f"    Std Score: {summary['std_score']:.6f}")
                print(f"    Mean Time: {summary['mean_time']:.6f}s")
                print(f"    Mean Effort: {summary['mean_effort']:.2f}")
                print(f"    Success Rate: {summary['success_rate']:.2%}")
            
            print(f"\n  Results saved to: {output_path}")
            
        except Exception as e:
            print(f"ERROR running {config_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"\nAll results saved in: {output_dir}/")
    
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
