"""
Generate visualization charts from existing result files.

This script reads result files from data/summary_results/ and generates:
- Performance comparison charts
- Stability analysis
- Efficiency analysis
- Convergence plots
- 3D surface plots (if problem data available)

Usage:
    python -m visualization.generate_charts_from_results
    python -m visualization.generate_charts_from_results --problem tsp
    python -m visualization.generate_charts_from_results --all
"""

import sys
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Import from within visualization package
from . import (
    plot_performance_bar,
    plot_stability_analysis,
    plot_computation_efficiency,
    parse_results_file,
    plot_tsp_objective_surface,
    plot_knapsack_objective_surface,
    plot_nqueens_objective_surface,
    plot_graph_coloring_objective_surface,
    create_3d_visualization_suite,
)


def parse_discrete_results(filepath: str, is_continuous: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Parse discrete algorithm results from text file.
    
    Args:
        filepath: Path to result file
        is_continuous: Whether this is a continuous problem (adds convergence metrics)
    
    Returns dict with structure:
    {
        'AlgorithmName': {
            'best_score': float,
            'mean_score': float,
            'std_score': float,
            'mean_time': float,
            'mean_effort': float,
            'success_rate': float,
            'runs': int,
            # For continuous problems only:
            'convergence_speed': float,      # Tốc độ hội tụ
            'convergence_capability': float,  # Khả năng hội tụ
        }
    }
    """
    results = {}
    current_algo = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by experiment sections
    sections = re.split(r'=+', content)
    
    for section in sections:
        if not section.strip():
            continue
            
        algo_match = re.search(r'Algorithm\s*:\s*(\w+)', section)
        if not algo_match:
            continue
            
        algo_name = algo_match.group(1)
        
        # Extract metrics
        metrics = {}
        
        # Runs
        runs_match = re.search(r'Runs\s*:\s*(\d+)', section)
        metrics['runs'] = int(runs_match.group(1)) if runs_match else 0
        
        # Best score
        best_match = re.search(r'Best score\s*:\s*([-\d.eE]+)', section)
        metrics['best_score'] = float(best_match.group(1)) if best_match else None
        
        # Mean score
        mean_match = re.search(r'Mean score\s*:\s*([-\d.eE]+)', section)
        metrics['mean_score'] = float(mean_match.group(1)) if mean_match else None
        
        # Std score
        std_match = re.search(r'Std score\s*:\s*([-\d.eE]+)', section)
        metrics['std_score'] = float(std_match.group(1)) if std_match else 0.0
        
        # Success rate
        success_match = re.search(r'Success rate\s*:\s*([\d.]+)%', section)
        if success_match:
            metrics['success_rate'] = float(success_match.group(1)) / 100
        else:
            metrics['success_rate'] = 0.0
        
        # Mean time
        time_match = re.search(r'Mean time \(s\)\s*:\s*([\d.eE+-]+)', section)
        metrics['mean_time'] = float(time_match.group(1)) if time_match else 0.0
        
        # Mean effort
        effort_match = re.search(r'Mean effort\s*:\s*([\d.eE+-]+)', section)
        metrics['mean_effort'] = float(effort_match.group(1)) if effort_match else 0.0
        
        # For continuous problems, calculate convergence metrics
        if is_continuous:
            # Convergence speed: success rate divided by time (higher = faster convergence)
            # Add small epsilon to avoid division by zero
            time_factor = metrics['mean_time'] + 0.001
            metrics['convergence_speed'] = metrics['success_rate'] / time_factor
            
            # Convergence capability: directly from success rate
            metrics['convergence_capability'] = metrics['success_rate']
        
        results[algo_name] = metrics
    
    return results


# List of continuous problems for automatic detection
CONTINUOUS_PROBLEMS = {'sphere', 'ackley', 'rastrigin', 'rosenbrock', 'griewank'}


def is_continuous_problem(problem_name: str) -> bool:
    """Check if a problem is continuous based on its name."""
    return problem_name.lower() in CONTINUOUS_PROBLEMS


def plot_convergence_metrics(
    results: Dict[str, Dict[str, Any]],
    problem_name: str,
    save_path: str
):
    """
    Plot convergence speed and capability for continuous problems.
    
    Args:
        results: Dictionary of algorithm results with convergence metrics
        problem_name: Name of the problem
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    algorithms = list(results.keys())
    convergence_speeds = [results[algo].get('convergence_speed', 0) for algo in algorithms]
    convergence_capabilities = [results[algo].get('convergence_capability', 0) for algo in algorithms]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Convergence Speed
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
    bars1 = ax1.bar(algorithms, convergence_speeds, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Convergence Speed (Success Rate / Time)', fontweight='bold')
    ax1.set_title(f'{problem_name}: Convergence Speed\n(Higher = Faster Convergence)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars1, convergence_speeds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Convergence Capability
    ax2 = axes[1]
    bars2 = ax2.bar(algorithms, convergence_capabilities, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Convergence Capability (Success Rate)', fontweight='bold')
    ax2.set_title(f'{problem_name}: Convergence Capability\n(Higher = Better Convergence)', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars2, convergence_capabilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - Convergence metrics plot saved: {save_path}")


def generate_problem_visualizations(
    problem_name: str,
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Generate all visualizations for a problem from results."""
    
    print(f"\n  Generating visualizations for {problem_name}...")
    
    # Create output subdirectories
    perf_dir = output_dir / problem_name / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance bar chart
    metrics = ['mean_score', 'best_score', 'std_score', 'mean_time', 'mean_effort']
    try:
        plot_performance_bar(
            results,
            metrics=metrics,
            title=f"{problem_name}: Multi-Metric Performance Comparison",
            save_path=str(perf_dir / "performance_bar.png"),
            normalize=True
        )
        plt.close()
        print("    - Performance bar chart")
    except Exception as e:
        print(f"    - Performance bar chart failed: {e}")
    
    # 2. Stability analysis
    try:
        plot_stability_analysis(
            results,
            title=f"{problem_name}: Stability Analysis",
            save_path=str(perf_dir / "stability_analysis.png")
        )
        plt.close()
        print("    - Stability analysis")
    except Exception as e:
        print(f"    - Stability analysis failed: {e}")
    
    # 3. Computational efficiency
    try:
        plot_computation_efficiency(
            results,
            title=f"{problem_name}: Computational Efficiency",
            save_path=str(perf_dir / "efficiency.png")
        )
        plt.close()
        print("    - Efficiency analysis")
    except Exception as e:
        print(f"    - Efficiency analysis failed: {e}")


def generate_3d_surface_from_problem(
    problem_name: str,
    output_dir: Path
):
    """Generate 3D surface plots if problem data is available."""
    
    surface_dir = output_dir / problem_name / "3d_surface"
    surface_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Generating 3D surface for {problem_name}...")
    
    try:
        if problem_name.lower() == 'tsp':
            # Try to load TSP data from existing files or generate sample
            import random
            random.seed(42)
            n_cities = 8
            distance_matrix = [[0 if i == j else random.randint(10, 100)
                               for j in range(n_cities)] for i in range(n_cities)]
            
            plot_tsp_objective_surface(
                np.array(distance_matrix),
                title=f"{problem_name}: Objective Function Landscape",
                save_path=str(surface_dir / "tsp_3d_surface.png")
            )
            plt.close()
            print("    - TSP 3D surface")
            
        elif problem_name.lower() == 'knapsack':
            # Generate sample knapsack data
            import random
            random.seed(42)
            weights = [random.randint(5, 50) for _ in range(15)]
            values = [random.randint(10, 100) for _ in range(15)]
            capacity = 150
            
            plot_knapsack_objective_surface(
                weights, values, capacity,
                sample_points=2000,
                title=f"{problem_name}: Objective Function Landscape",
                save_path=str(surface_dir / "knapsack_3d_surface.png")
            )
            plt.close()
            print("    - Knapsack 3D surface")
            
        elif problem_name.lower() in ['n_queens', 'nqueens']:
            plot_nqueens_objective_surface(
                n=8,
                sample_points=3000,
                title=f"{problem_name}: Objective Function Landscape",
                save_path=str(surface_dir / "nqueens_3d_surface.png")
            )
            plt.close()
            print("    - N-Queens 3D surface")
        
        elif problem_name.lower() == 'graph_coloring':
            # Generate sample graph coloring data
            import random
            random.seed(42)
            n_nodes = 6
            # Create a simple graph adjacency matrix
            adjacency_matrix = [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 1, 0, 1, 1, 0],
                [0, 1, 1, 0, 1, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 0]
            ]
            num_colors = 3
            
            plot_graph_coloring_objective_surface(
                adjacency_matrix, num_colors,
                sample_points=3000,
                title=f"{problem_name}: Objective Function Landscape",
                save_path=str(surface_dir / "graph_coloring_3d_surface.png")
            )
            plt.close()
            print("    - Graph Coloring 3D surface")
            
    except Exception as e:
        print(f"    - 3D surface failed: {e}")


def generate_summary_report(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path
):
    """Generate comprehensive summary reports in both Markdown and TXT table format."""
    
    # 1. Generate Markdown report (for viewing)
    md_path = output_dir / "summary_report.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Discrete Optimization Algorithm Visualization Report\n\n")
        f.write(f"Generated from existing result files\n\n")
        f.write("## Summary\n\n")
        
        for problem_name, results in all_results.items():
            f.write(f"\n### {problem_name}\n\n")
            f.write("| Algorithm | Best Score | Mean Score | Std Dev | Time (s) | Success Rate |\n")
            f.write("|-----------|------------|------------|---------|----------|--------------|\n")
            
            for algo_name, metrics in results.items():
                best = f"{metrics['best_score']:.2f}" if metrics['best_score'] is not None else "N/A"
                mean = f"{metrics['mean_score']:.2f}" if metrics['mean_score'] is not None else "N/A"
                std = f"{metrics['std_score']:.2f}"
                time_str = f"{metrics['mean_time']:.4f}"
                success = f"{metrics['success_rate']:.1%}"
                
                f.write(f"| {algo_name} | {best} | {mean} | {std} | {time_str} | {success} |\n")
            
            f.write("\n")
    
    print(f"\n  Markdown report saved to: {md_path}")
    
    # 2. Generate TXT table report for each problem (formatted as table)
    for problem_name, results in all_results.items():
        txt_path = output_dir / f"{problem_name.lower()}_summary_table.txt"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"SUMMARY TABLE: {problem_name}\n")
            f.write("="*100 + "\n\n")
            
            # Define column headers and widths
            headers = ["Algorithm", "Best Score", "Mean Score", "Std Dev", "Time (s)", "Effort", "Success Rate"]
            widths = [18, 14, 14, 12, 12, 12, 14]
            
            # Write header row
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
            f.write(header_line + "\n")
            f.write("-" * len(header_line) + "\n")
            
            # Write data rows (each row = one algorithm)
            for algo_name, metrics in results.items():
                best = f"{metrics['best_score']:.4f}" if metrics['best_score'] is not None else "N/A"
                mean = f"{metrics['mean_score']:.4f}" if metrics['mean_score'] is not None else "N/A"
                std = f"{metrics['std_score']:.4f}"
                time_str = f"{metrics['mean_time']:.6f}"
                effort = f"{metrics['mean_effort']:.1f}"
                success = f"{metrics['success_rate']:.2%}"
                
                values = [algo_name, best, mean, std, time_str, effort, success]
                row_line = " | ".join(str(v).ljust(w) for v, w in zip(values, widths))
                f.write(row_line + "\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("Columns: Algorithm | Best Score | Mean Score | Std Dev | Time (s) | Effort | Success Rate\n")
            f.write("Note: Each row represents one algorithm with all its metrics\n")
        
        print(f"  Table summary saved to: {txt_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations from existing result files'
    )
    parser.add_argument(
        '--problem',
        help='Specific problem to visualize (e.g., tsp, knapsack)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available result files'
    )
    parser.add_argument(
        '--output',
        default='data/charts_from_results',
        help='Output directory (default: data/charts_from_results)'
    )
    
    args = parser.parse_args()
    
    # Find result files (relative to project root)
    results_dir = Path("data/summary_results")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Make sure you have run experiments first.")
        return
    
    # Determine which files to process
    if args.problem:
        result_files = [results_dir / f"{args.problem}_results.txt"]
    else:
        result_files = list(results_dir.glob("*_results.txt"))
    
    if not result_files:
        print("No result files found!")
        return
    
    # Create output directory (default: data/charts_from_results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS FROM RESULT FILES")
    print("="*70)
    print(f"Output Directory: {output_dir}")
    print(f"Result Files Found: {len(result_files)}")
    print("="*70)
    
    all_results = {}
    
    # Process each result file
    for result_file in result_files:
        if not result_file.exists():
            print(f"\nSkipping {result_file} (not found)")
            continue
        
        problem_name = result_file.stem.replace('_results', '').upper()
        
        print(f"\nProcessing: {result_file.name}")
        
        try:
            # Check if this is a continuous problem
            continuous = is_continuous_problem(problem_name)
            
            # Parse results with convergence metrics for continuous problems
            results = parse_discrete_results(str(result_file), is_continuous=continuous)
            
            if not results:
                print(f"  No valid results found in {result_file}")
                continue
            
            print(f"  Found {len(results)} algorithms")
            if continuous:
                print(f"  [Continuous Problem - Convergence metrics enabled]")
            
            # Store results
            all_results[problem_name] = results
            
            # Generate visualizations
            generate_problem_visualizations(problem_name, results, output_dir)
            
            # For continuous problems, generate convergence metrics
            if continuous:
                try:
                    perf_dir = output_dir / problem_name / "performance"
                    plot_convergence_metrics(
                        results,
                        problem_name,
                        save_path=str(perf_dir / "convergence_metrics.png")
                    )
                except Exception as e:
                    print(f"    - Convergence metrics failed: {e}")
            
            # Generate 3D surface
            generate_3d_surface_from_problem(problem_name, output_dir)
            
        except Exception as e:
            print(f"  Error processing {result_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    if all_results:
        generate_summary_report(all_results, output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*70)
    print(f"\nAll charts saved to: {output_dir}")
    print("\nDirectory structure:")
    for problem in all_results.keys():
        print(f"  {output_dir}/{problem}/")
        print(f"    - performance/     # Bar charts, stability, efficiency")
        print(f"    - 3d_surface/      # 3D objective surface plots")
    print(f"\n  {output_dir}/summary_report.md")


if __name__ == "__main__":
    main()
