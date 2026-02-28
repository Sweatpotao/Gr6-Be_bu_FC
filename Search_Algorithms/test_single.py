"""
CLI tool to test a single problem with selected algorithms.

Examples:
    # Test N-Queens with config from YAML (auto-detect config/n_queens.yaml)
    python test_single.py --problem n_queens

    # Test with specific config file
    python test_single.py --config config/n_queens.yaml --algorithms BFS,AStar

    # Test with specific algorithms (auto-detect YAML or use default params)
    python test_single.py --problem n_queens --algorithms BFS,AStar

    # Test TSP with metaheuristic algorithms
    python test_single.py --problem tsp --algorithms GA_TSP,HillClimbingTSP,SimulatedAnnealingTSP

    # Test Knapsack with metaheuristic algorithms
    python test_single.py --problem knapsack --algorithms ABC_Knapsack,TLBO_Knapsack

    # Test with 10 runs
    python test_single.py --problem tsp --runs 10

    # Test Knapsack and save results
    python test_single.py --problem knapsack --algorithms Greedy,UCS --output results.txt

Note: Problem-specific algorithms (e.g., GA_TSP for TSP, ABC_Knapsack for Knapsack)
    will show a warning if selected for incompatible problems.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from experiment.run_experiment import run_experiment_from_dict
from experiment.registry import PROBLEM_REGISTRY, ALGORITHM_REGISTRY

# Define problem-specific algorithm compatibility
# Algorithms that only work with specific problems
PROBLEM_SPECIFIC_ALGORITHMS = {
    'tsp': ['GA_TSP', 'HillClimbingTSP', 'SimulatedAnnealingTSP'],
    'knapsack': ['ABC_Knapsack', 'TLBO_Knapsack'],
}

# Reverse mapping: algorithm -> compatible problems
ALGORITHM_COMPATIBILITY = {}
for problem, algos in PROBLEM_SPECIFIC_ALGORITHMS.items():
    for algo in algos:
        ALGORITHM_COMPATIBILITY[algo] = [problem]

def check_algorithm_compatibility(algorithms, problem_name):
    """
    Check if selected algorithms are compatible with the problem.
    
    Returns:
        tuple: (valid_algorithms, warnings)
    """
    valid_algos = []
    warnings = []
    
    for algo in algorithms:
        # Check if algorithm is problem-specific
        compatible_problems = ALGORITHM_COMPATIBILITY.get(algo)
        
        if compatible_problems is not None:
            # This is a problem-specific algorithm
            if problem_name not in compatible_problems:
                warnings.append(
                    f"WARNING: '{algo}' is designed for {compatible_problems[0]} problems, "
                    f"not for '{problem_name}'. It may not work correctly or produce invalid results."
                )
            else:
                valid_algos.append(algo)
        else:
            # General algorithm, should work with any problem
            valid_algos.append(algo)
    
    return valid_algos, warnings


def find_config_file(problem_name):
    """Find YAML config file for a problem in config directory."""
    config_dir = Path(__file__).parent / "config"
    
    # Try common naming patterns
    possible_names = [
        f"{problem_name}.yaml",
        f"{problem_name}_experiment.yaml",
        f"{problem_name}_config.yaml",
    ]
    
    for name in possible_names:
        config_path = config_dir / name
        if config_path.exists():
            return str(config_path)
    
    return None


def load_config_from_yaml(config_path):
    """Load experiment config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_problem_params(problem_name):
    """Get default parameters for each problem type."""
    defaults = {
        "n_queens": {"n": 8},
        "tsp": {
            "distance_matrix": [
                [0, 10, 15, 20],
                [10, 0, 35, 25],
                [15, 35, 0, 30],
                [20, 25, 30, 0]
            ],
            "start_city": 0
        },
        "knapsack": {
            "values": [60, 100, 120],
            "weights": [10, 20, 30],
            "capacity": 50
        },
        "graph_coloring": {
            "adjacency_matrix": [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0]
            ],
            "num_colors": 3
        },
        "grid_pathfinding": {
            "grid": [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0]
            ],
            "start": (0, 0),
            "goal": (3, 3)
        },
        # Continuous problems
        "sphere": {"dim": 10},
        "ackley": {"dim": 10},
        "rastrigin": {"dim": 10},
        "rosenbrock": {"dim": 10},
        "griewank": {"dim": 10},
    }
    return defaults.get(problem_name, {})


def build_config(problem_name, algorithms, runs, timeout, config_path=None):
    """Build config dict from CLI arguments or YAML file.
    
    Priority:
    1. If config_path provided, load from YAML
    2. If auto-detect finds YAML, load from YAML
    3. Otherwise, use default params
    """
    # Try to find/load YAML config
    yaml_config = None
    if config_path:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        yaml_config = load_config_from_yaml(config_path)
        print(f"  Config:      {config_path}")
    else:
        # Auto-detect config file
        auto_config_path = find_config_file(problem_name)
        if auto_config_path:
            yaml_config = load_config_from_yaml(auto_config_path)
            print(f"  Config:      {auto_config_path} (auto-detected)")
    
    if yaml_config:
        # Use YAML config as base
        config = yaml_config.copy()
        
        # Override with CLI arguments if provided
        if runs != 5:  # Default value changed
            config["experiment"]["runs"] = runs
        if algorithms is not None:
            # Filter algorithms from YAML or use specified ones
            yaml_algos = config.get("algorithms", {})
            filtered_algos = {}
            for algo in algorithms:
                if algo in yaml_algos:
                    filtered_algos[algo] = yaml_algos[algo]
                else:
                    # Use default config for algorithms not in YAML
                    filtered_algos[algo] = {"timeout": timeout}
                    if algo == "DFS":
                        filtered_algos[algo]["depth_limit"] = 1000
            config["algorithms"] = filtered_algos
        # If no algorithms specified in CLI, use all from YAML
        if timeout != 60:  # Default timeout changed
            for algo_cfg in config["algorithms"].values():
                algo_cfg["timeout"] = timeout
        
        return config
    
    # Fallback: Build from default params
    print(f"  Config:      Default params (no YAML found)")
    problem_params = get_default_problem_params(problem_name)
    
    # Build algorithm configs
    algos_to_use = algorithms if algorithms else list(ALGORITHM_REGISTRY.keys())
    algo_configs = {}
    for algo in algos_to_use:
        algo_configs[algo] = {"timeout": timeout}
        if algo == "DFS":
            algo_configs[algo]["depth_limit"] = 1000
    
    config = {
        "experiment": {"runs": runs},
        "problem": {
            "name": problem_name,
            "params": problem_params
        },
        "algorithms": algo_configs
    }
    
    return config


def print_results(results, problem_name):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print(f"RESULTS FOR: {problem_name.upper()}")
    print("=" * 70)
    
    for algo_name, summary in results.items():
        print(f"\n  Algorithm: {algo_name}")
        print(f"  {'-' * 50}")
        print(f"    Runs:          {summary['runs']}")
        
        if summary['best_score'] is not None:
            print(f"    Best Score:    {summary['best_score']:.6f}")
            print(f"    Mean Score:    {summary['mean_score']:.6f}")
            print(f"    Std Score:     {summary['std_score']:.6f}")
        else:
            print(f"    Best Score:    N/A")
            print(f"    Mean Score:    N/A")
            print(f"    Std Score:     N/A")
        
        print(f"    Mean Time:     {summary['mean_time']:.6f}s")
        print(f"    Mean Effort:   {summary['mean_effort']:.2f}")
        print(f"    Success Rate:  {summary['success_rate']:.2%}")
        
        if summary.get('timeout_count', 0) > 0:
            print(f"    Timeouts:      {summary['timeout_count']}")


def save_results(results, problem_name, output_path):
    """Save results to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"TEST RESULTS - {problem_name.upper()}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        
        for algo_name, summary in results.items():
            f.write(f"Algorithm: {algo_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Runs:          {summary['runs']}\n")
            
            if summary['best_score'] is not None:
                f.write(f"  Best Score:    {summary['best_score']:.6f}\n")
                f.write(f"  Mean Score:    {summary['mean_score']:.6f}\n")
                f.write(f"  Std Score:     {summary['std_score']:.6f}\n")
            else:
                f.write(f"  Best Score:    N/A\n")
                f.write(f"  Mean Score:    N/A\n")
                f.write(f"  Std Score:     N/A\n")
            
            f.write(f"  Mean Time:     {summary['mean_time']:.6f}s\n")
            f.write(f"  Mean Effort:   {summary['mean_effort']:.2f}\n")
            f.write(f"  Success Rate:  {summary['success_rate']:.2%}\n")
            
            if summary.get('timeout_count', 0) > 0:
                f.write(f"  Timeouts:      {summary['timeout_count']}\n")
            
            f.write("\n")
    
    print(f"\nResults saved to: {output_path}")


def convert_to_spider_data(results, problem_name):
    """
    Convert test results to spider chart format for visualization.
    
    Returns dict in format: {problem_name: {algo_name: [quality, speed, efficiency, reliability, convergence]}}
    All values are normalized to 0-1 scale.
    """
    spider_data = {problem_name: {}}
    
    if not results:
        return spider_data
    
    # Find max values for normalization
    max_time = max(r['mean_time'] for r in results.values() if r['mean_time'] > 0) or 1.0
    max_effort = max(r['mean_effort'] for r in results.values() if r['mean_effort'] > 0) or 1.0
    
    for algo_name, summary in results.items():
        # Normalize metrics to 0-1 scale (higher is better)
        # Solution Quality: based on success rate and best score
        if summary['best_score'] is not None:
            quality = summary['success_rate'] * 0.5 + 0.5  # Scale success rate
        else:
            quality = summary['success_rate']
        
        # Speed: inverse of time (faster is better)
        speed = 1 - (summary['mean_time'] / max_time) if max_time > 0 else 0
        
        # Efficiency: inverse of effort (fewer nodes is better)
        efficiency = 1 - (summary['mean_effort'] / max_effort) if max_effort > 0 else 0
        
        # Reliability: success rate
        reliability = summary['success_rate']
        
        # Convergence: same as success rate for discrete problems
        convergence = summary['success_rate']
        
        spider_data[problem_name][algo_name] = [
            round(quality, 4),
            round(speed, 4),
            round(efficiency, 4),
            round(reliability, 4),
            round(convergence, 4)
        ]
    
    return spider_data


def save_spider_json(results, problem_name, output_path=None):
    """
    Save results in spider chart JSON format for visualization.
    
    Args:
        results: Raw experiment results
        problem_name: Name of the problem
        output_path: Optional path for JSON file (default: {problem_name}_spider_data.json)
    """
    import json
    
    # Generate default filename if not provided
    if output_path is None:
        safe_name = problem_name.replace(" ", "_").lower()
        output_path = f"{safe_name}_spider_data.json"
    
    # Convert to spider format
    spider_data = convert_to_spider_data(results, problem_name)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(spider_data, f, indent=2)
    
    print(f"Spider chart data saved to: {output_path}")
    print(f"  Use: python -m visualization.discrete_comparison {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Test a single problem with selected algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect config file (config/n_queens.yaml)
  python test_single.py --problem n_queens

  # Use specific config file
  python test_single.py --config config/n_queens.yaml --algorithms BFS,AStar

  # Use default params (no YAML)
  python test_single.py --problem n_queens --algorithms BFS,AStar

  # Override YAML settings
  python test_single.py --problem tsp --runs 10 --timeout 30
  python test_single.py --problem knapsack --algorithms Greedy --output results.txt
        """
    )
    
    parser.add_argument(
        '--problem', 
        default=None,
        help='Problem name (n_queens, tsp, knapsack, grid_pathfinding, sphere, ackley, ...)'
    )
    parser.add_argument(
        '--config', 
        default=None,
        help='Path to YAML config file (optional, auto-detected if not specified)'
    )
    parser.add_argument(
        '--algorithms', 
        default=None, 
        help='Comma-separated algorithm names (default: from YAML or all available)'
    )
    parser.add_argument(
        '--runs', 
        type=int, 
        default=5, 
        help='Number of runs (default: 5, overrides YAML if specified)'
    )
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=60, 
        help='Timeout per run in seconds (default: 60, overrides YAML if specified)'
    )
    parser.add_argument(
        '--output', 
        default=None, 
        help='Output file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Determine problem name
    problem_name = args.problem
    
    # If --config specified but no --problem, extract problem from config
    if args.config and not problem_name:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        yaml_config = load_config_from_yaml(args.config)
        problem_name = yaml_config.get("problem", {}).get("name")
        if not problem_name:
            print("Error: Could not determine problem name from config file")
            print("Please specify --problem explicitly")
            sys.exit(1)
    
    # Must have problem name at this point
    if not problem_name:
        print("Error: Must specify --problem or --config")
        parser.print_help()
        sys.exit(1)
    
    # Validate problem
    if problem_name not in PROBLEM_REGISTRY:
        print(f"Error: Unknown problem '{problem_name}'")
        print(f"\nAvailable problems:")
        for name in sorted(PROBLEM_REGISTRY.keys()):
            print(f"  - {name}")
        sys.exit(1)
    
    # Parse and validate algorithms
    if args.algorithms:
        selected_algos = [a.strip() for a in args.algorithms.split(',')]
        invalid_algos = []
        for algo in selected_algos:
            if algo not in ALGORITHM_REGISTRY:
                invalid_algos.append(algo)
        
        if invalid_algos:
            print(f"Error: Unknown algorithm(s): {', '.join(invalid_algos)}")
            print(f"\nAvailable algorithms:")
            for name in sorted(ALGORITHM_REGISTRY.keys()):
                print(f"  - {name}")
            sys.exit(1)
        
        # Check algorithm-problem compatibility
        valid_algos, warnings = check_algorithm_compatibility(selected_algos, problem_name)
        
        # Print warnings for incompatible algorithms
        if warnings:
            print("\n" + "!" * 70)
            for warning in warnings:
                print(f"  {warning}")
            print("!" * 70 + "\n")
            
            # Ask user if they want to continue (only for interactive use)
            # In non-interactive mode, we'll continue but skip incompatible algorithms
            
        # Filter to only valid algorithms for this problem
        selected_algos = valid_algos if valid_algos else None
        
    else:
        selected_algos = None  # Will be determined from YAML or use all
    
    # Build config
    config = build_config(problem_name, selected_algos, args.runs, args.timeout, args.config)
    
    # Get final algorithm list from config
    final_algos = list(config["algorithms"].keys())
    
    # Print test info
    print("\n" + "=" * 70)
    print(f"TEST CONFIGURATION")
    print("=" * 70)
    print(f"  Problem:     {problem_name}")
    print(f"  Algorithms:  {', '.join(final_algos)}")
    print(f"  Runs:        {config['experiment']['runs']}")
    print(f"  Timeout:     {args.timeout}s")
    print("=" * 70)
    
    # Run experiment
    try:
        results = run_experiment_from_dict(config)
        
        # Print results
        print_results(results, problem_name)
        
        # Save if output specified
        if args.output:
            save_results(results, problem_name, args.output)
        
        # Automatically save spider chart JSON for visualization
        spider_json_path = save_spider_json(results, problem_name)
        
        # Final summary
        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nTo visualize results, run:")
        print(f"  python -m visualization.discrete_comparison {spider_json_path}")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
