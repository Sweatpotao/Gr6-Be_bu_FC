"""
Run all experiments for continuous and discrete problems.
Each problem runs 4 times with available algorithms.
"""

from experiment.run_experiment import run_experiment
from experiment.logger import save_summary_txt
from visualization.continuous_comparison import plot_radar_chart
from visualization.discrete_comparison import create_spider_chart
from visualization import (
    parse_discrete_results,
    is_continuous_problem,
    generate_problem_visualizations,
    generate_3d_surface_from_problem,
    generate_summary_report,
)
from visualization.sensitivity_analysis import ParameterSensitivityAnalyzer
from experiment.registry import PROBLEM_REGISTRY, ALGORITHM_REGISTRY
import yaml
import copy
import os
from visualization.plot_comparison import plot_continuous_radar
from datetime import datetime
from pathlib import Path

# Discrete problems list
DISCRETE_PROBLEMS = {'grid_pathfinding', 'n_queens', 'tsp', 'knapsack', 'graph_coloring'}


def save_results_as_table(results: dict, output_path: str, problem_name: str = ""):
    """
    Save experiment results as a formatted table.
    Each row represents one algorithm, each column represents a metric.
    
    Args:
        results: Dictionary of {algorithm_name: metrics_dict}
        output_path: Path to output file
        problem_name: Name of the problem (for header)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 110 + "\n")
        if problem_name:
            f.write(f"RESULTS FOR: {problem_name.upper()}\n")
        else:
            f.write("EXPERIMENT RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 110 + "\n\n")
        
        # Define column headers and widths
        headers = ["Algorithm", "Best Score", "Mean Score", "Std Dev", "Time (s)", "Effort", "Success Rate", "Runs"]
        widths = [18, 14, 14, 12, 12, 10, 14, 8]
        
        # Write header row
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        
        # Write data rows - each algorithm on one row
        for algo_name, summary in results.items():
            best = f"{summary['best_score']:.6f}" if summary['best_score'] is not None else "N/A"
            mean = f"{summary['mean_score']:.6f}" if summary['mean_score'] is not None else "N/A"
            std = f"{summary['std_score']:.6f}"
            time_str = f"{summary['mean_time']:.6f}"
            effort = f"{summary['mean_effort']:.1f}"
            success = f"{summary['success_rate']:.2%}"
            runs = f"{summary['runs']}"
            
            values = [algo_name, best, mean, std, time_str, effort, success, runs]
            row_line = " | ".join(str(v).ljust(w) for v, w in zip(values, widths))
            f.write(row_line + "\n")
        
        f.write("\n" + "=" * 110 + "\n")
        f.write("Note: Each row represents one algorithm. Columns are the evaluation metrics.\n")
        f.write("      Best Score = Best solution found | Mean Score = Average solution quality\n")
        f.write("      Std Dev = Standard deviation | Time (s) = Average runtime in seconds\n")
        f.write("      Effort = Average number of evaluations | Success Rate = Percentage of successful runs\n")
    
    print(f"  [+] Saved table format: {output_path}")


def run_sensitivity_analysis(problem_name: str, config_path: str, results: dict, output_dir: str = "data/sensitivity_analysis"):
    """
    Run parameter sensitivity analysis for metaheuristic algorithms.
    Only analyzes the best performing metaheuristic algorithm for each problem.
    
    Args:
        problem_name: Name of the problem
        config_path: Path to the YAML config file
        results: Experiment results dict {algorithm_name: metrics}
        output_dir: Directory to save sensitivity analysis results
    """
    # Metaheuristic algorithms that have tunable parameters
    METAHEURISTIC_ALGOS = {
        'GA_TSP', 'GA', 'DE', 'PSO', 'ACO', 'ACO_Discrete',
        'ABC', 'ABC_Knapsack', 'TLBO', 'TLBO_Knapsack',
        'CuckooSearch', 'FireflyAlgorithm',
        'SimulatedAnnealing', 'SimulatedAnnealingTSP',
        'HillClimbing', 'HillClimbingTSP'
    }
    
    # Filter results to only include metaheuristics
    meta_results = {k: v for k, v in results.items() if k in METAHEURISTIC_ALGOS}
    
    if not meta_results:
        print(f"  [!] No metaheuristic algorithms found for sensitivity analysis")
        return
    
    # Find best metaheuristic algorithm (by best_score, lower is better)
    best_algo = min(meta_results.items(), key=lambda x: x[1].get('best_score', float('inf')) if x[1].get('best_score') is not None else float('inf'))[0]
    
    print(f"\n  [{'='*68}]")
    print(f"  Running Sensitivity Analysis for {best_algo} on {problem_name}")
    print(f"  [{'='*68}]")
    
    try:
        # Load config to get problem parameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create problem instance
        prob_name = config['problem']['name']
        prob_params = config['problem']['params']
        ProblemClass = PROBLEM_REGISTRY[prob_name]
        problem = ProblemClass(**prob_params)
        
        # Get algorithm class
        AlgoClass = ALGORITHM_REGISTRY[best_algo]
        
        # Define parameter ranges for sensitivity analysis
        # These are common parameters for metaheuristic algorithms
        param_configs = {
            'GA_TSP': {
                'pop_size': [20, 50, 100],
                'mutation_rate': [0.05, 0.1, 0.2]
            },
            'GA': {
                'pop_size': [20, 50, 100],
                'mutation_rate': [0.05, 0.1, 0.2]
            },
            'DE': {
                'pop_size': [20, 50, 100],
                'F': [0.5, 0.8, 1.0]
            },
            'PSO': {
                'pop_size': [20, 50, 100],
                'c1': [1.0, 2.0, 2.5]
            },
            'ACO_Discrete': {
                'n_ants': [20, 50, 100],
                'alpha': [0.5, 1.0, 2.0]
            },
            'ABC': {
                'pop_size': [20, 50, 100],
                'limit': [50, 100, 200]
            },
            'ABC_Knapsack': {
                'pop_size': [20, 50, 100],
                'limit': [50, 100, 200]
            },
            'TLBO': {
                'pop_size': [20, 50, 100],
            },
            'TLBO_Knapsack': {
                'pop_size': [20, 50, 100],
            },
            'SimulatedAnnealingTSP': {
                'initial_temp': [500, 1000, 2000],
            },
            'SimulatedAnnealing': {
                'initial_temp': [500, 1000, 2000],
            }
        }
        
        # Get default algo config from YAML
        algo_config = config.get('algorithms', {}).get(best_algo, {})
        
        # Create analyzer
        analyzer = ParameterSensitivityAnalyzer(AlgoClass, problem)
        
        # Run 1D sensitivity analysis for available parameters
        param_ranges = param_configs.get(best_algo, {})
        
        if param_ranges:
            for param_name, param_values in param_ranges.items():
                print(f"    Analyzing {param_name}: {param_values}")
                try:
                    analyzer.analyze_1d(
                        param_name, 
                        param_values, 
                        fixed_params=algo_config,
                        n_runs=3  # Reduced runs for faster execution
                    )
                except Exception as e:
                    print(f"      [!] Failed to analyze {param_name}: {e}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        analyzer.plot_results(output_dir=os.path.join(output_dir, problem_name))
        print(f"  [+] Sensitivity analysis saved to: {output_dir}/{problem_name}/")
        
    except Exception as e:
        print(f"  [!] Sensitivity analysis failed: {e}")
        import traceback
        traceback.print_exc()


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


def generate_charts_from_experiments(all_results: dict, output_dir: str = "data/charts_from_results"):
    """
    Automatically generate visualization charts from experiment results.
    
    Args:
        all_results: Dictionary of {config_path: results} from experiments
        output_dir: Directory to save generated charts
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION CHARTS")
    print("=" * 70)
    print(f"Output Directory: {output_path}")
    
    all_parsed_results = {}
    
    for config_path, results in all_results.items():
        problem_name = config_path.split('/')[-1].replace('_experiment.yaml', '').replace('.yaml', '')
        problem_name_upper = problem_name.upper()
        
        print(f"\n  Processing: {problem_name}")
        
        # Convert results to the format expected by visualization functions
        parsed_results = {}
        for algo_name, summary in results.items():
            parsed_results[algo_name] = {
                'best_score': summary.get('best_score'),
                'mean_score': summary.get('mean_score'),
                'std_score': summary.get('std_score', 0.0),
                'mean_time': summary.get('mean_time', 0.0),
                'mean_effort': summary.get('mean_effort', 0.0),
                'success_rate': summary.get('success_rate', 0.0),
                'runs': summary.get('runs', 0),
            }
        
        all_parsed_results[problem_name_upper] = parsed_results
        
        # Generate visualizations for this problem
        try:
            generate_problem_visualizations(problem_name_upper, parsed_results, output_path)
            print(f"    [+] Charts generated for {problem_name}")
        except Exception as e:
            print(f"    [!] Failed to generate charts for {problem_name}: {e}")
        
        # Generate 3D surface plots for specific problems
        try:
            generate_3d_surface_from_problem(problem_name_upper, output_path)
        except Exception as e:
            print(f"    [!] Failed to generate 3D surface for {problem_name}: {e}")
    
    # Generate summary report
    try:
        generate_summary_report(all_parsed_results, output_path)
        print(f"\n  [+] Summary report saved to: {output_path}/summary_report.md")
    except Exception as e:
        print(f"\n  [!] Failed to generate summary report: {e}")
    
    print("\n" + "=" * 70)
    print("CHART GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nAll charts saved to: {output_path}")

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
            
            # Save txt as table format (each row = one algorithm, each column = one metric)
            output_path = os.path.join(txt_dir, output_file)
            save_results_as_table(results, output_path, problem_name)
            
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
            if is_discrete_problem(problem_name):
                # Use discrete spider chart
                spider_data = convert_results_to_spider_data(results)
                chart_path = os.path.join(plots_dir, f"{problem_name}_spider.png")
                create_spider_chart(spider_data, title=f"{problem_name.upper()} - Algorithm Comparison", save_path=chart_path)
                print(f"  [+] Saved discrete chart: {chart_path}")
            else:
                # Use continuous radar chart
                plot_path = os.path.join(plots_dir, f"{problem_name}_radar.png")
                plot_continuous_radar(problem_name, results, plot_path)
                print(f"    [+] Saved continuous chart: {plot_path}")
            
            # Run sensitivity analysis for metaheuristic algorithms
            run_sensitivity_analysis(problem_name, config_path, results)
            
        except Exception as e:
            print(f"ERROR running {config_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"\nAll results saved in: {txt_dir}/ and {plots_dir}/")
    
    # Print final summary table
    print("\n" + "=" * 95)
    print("FINAL SUMMARY TABLE")
    print("=" * 95)
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
    
    # Automatically generate visualization charts from results
    generate_charts_from_experiments(all_results)

if __name__ == "__main__":
    main()