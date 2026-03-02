"""
Visualization module for discrete algorithm comparison results.
Creates spider/radar charts for comparing classical search algorithms (BFS, DFS, UCS, Greedy, A*).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    DISCRETE_COLOR_MAP, 
    calculate_angles, 
    ensure_output_dir,
    save_or_show
)


def create_spider_chart(algorithms_data: Dict[str, List[float]], 
                        title: str = "Algorithm Comparison",
                        save_path: Optional[str] = None):
    """
    Create a spider/radar chart for algorithm comparison.
    
    Args:
        algorithms_data: Dict mapping algorithm name to [quality, speed, efficiency, reliability, convergence]
        title: Chart title
        save_path: Path to save the figure (if None, display only)
    """
    categories = ['Solution Quality', 'Speed', 'Efficiency', 'Reliability', 'Convergence']
    N = len(categories)
    
    # Compute angle for each axis
    angles = calculate_angles(N)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each algorithm
    for idx, (name, values) in enumerate(algorithms_data.items()):
        values_closed = list(values) + values[:1]  # Complete the circle (create copy)
        color = DISCRETE_COLOR_MAP.get(name, plt.cm.Set2(idx / len(algorithms_data)))
        
        ax.plot(angles, values_closed, 'o-', linewidth=2.5, label=name, color=color, markersize=8)
        ax.fill(angles, values_closed, alpha=0.15, color=color)
    
    # Configure chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.set_title(title, size=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_or_show(fig, save_path)


def create_comparison_grid(spider_data: Dict, output_dir: str = "comparison_charts"):
    """
    Create spider charts for all problems in a grid layout.
    
    Args:
        spider_data: Dict mapping problem names to algorithm performance data
        output_dir: Directory to save charts
    """
    output_path = ensure_output_dir(output_dir)
    
    problems = list(spider_data.keys())
    n_problems = len(problems)
    
    # Calculate grid dimensions
    n_cols = min(2, n_problems)
    n_rows = (n_problems + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig = plt.figure(figsize=(n_cols * 8, n_rows * 8))
    
    categories = ['Solution Quality', 'Speed', 'Efficiency', 'Reliability', 'Convergence']
    N = len(categories)
    angles = calculate_angles(N)
    
    for idx, problem_name in enumerate(problems):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='polar')
        
        algorithms_data = spider_data[problem_name]
        
        for algo_idx, (algo_name, values) in enumerate(algorithms_data.items()):
            values_list = list(values)
            values_closed = values_list + values_list[:1]  # Complete the circle properly
            color = DISCRETE_COLOR_MAP.get(algo_name, plt.cm.Set2(algo_idx / len(algorithms_data)))
            ax.plot(angles, values_closed, 'o-', linewidth=2, label=algo_name, color=color)
            ax.fill(angles, values_closed, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_title(problem_name, size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Discrete Algorithm Comparison - All Problems', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_path / "discrete_comparison_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Grid chart saved to: {output_file}")
    plt.close()


def create_summary_table(spider_data: Dict, output_file: str = "discrete_summary.txt"):
    """
    Create a text summary table of algorithm rankings.
    
    Args:
        spider_data: Dict mapping problem names to algorithm performance data
        output_file: Path to save summary
    """
    summary = []
    summary.append("=" * 80)
    summary.append("DISCRETE ALGORITHM COMPARISON SUMMARY")
    summary.append("=" * 80)
    
    categories = ['Solution Quality', 'Speed', 'Efficiency', 'Reliability', 'Convergence']
    
    for problem_name, algorithms_data in spider_data.items():
        summary.append(f"\n{problem_name}")
        summary.append("-" * 80)
        
        # Calculate overall score (average of all metrics)
        rankings = []
        for algo_name, values in algorithms_data.items():
            overall = sum(values) / len(values)
            rankings.append((algo_name, overall, values))
        
        # Sort by overall score
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        summary.append(f"{'Rank':<6} {'Algorithm':<12} {'Overall':<10} {'Quality':<10} {'Speed':<8} {'Efficiency':<12} {'Reliability':<12} {'Convergence':<12}")
        summary.append("-" * 80)
        
        for rank, (algo_name, overall, values) in enumerate(rankings, 1):
            summary.append(
                f"{rank:<6} {algo_name:<12} {overall:>8.3f}  "
                f"{values[0]:>8.3f}  {values[1]:>6.3f}  {values[2]:>10.3f}  "
                f"{values[3]:>10.3f}  {values[4]:>10.3f}"
            )
        
        # Add winner
        winner = rankings[0][0]
        summary.append(f"\nWinner: {winner}")
    
    summary_text = "\n".join(summary)
    
    # Handle both string path and Path object
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nSummary saved to: {output_file}")


def main():
    """Main entry point for visualization."""
    import sys
    
    # Check if data file provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "discrete_spider_data.json"
    
    # Load data - handle both absolute and relative paths
    data_path = Path(data_file)
    if not data_path.is_absolute():
        # Try relative to current directory first
        if not data_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.parent
            data_path = script_dir / data_file
    
    if not data_path.exists():
        print(f"Error: Data file '{data_file}' not found.")
        print("Please run 'run_discrete_comparison.py' first to generate comparison data.")
        return
    
    with open(data_path, 'r') as f:
        spider_data = json.load(f)
    
    print("Creating visualizations...")
    
    # Create output directory
    output_dir = Path("comparison_charts")
    output_dir.mkdir(exist_ok=True)
    
    # Create individual charts for each problem
    for problem_name, algorithms_data in spider_data.items():
        safe_name = problem_name.replace(" ", "_").lower()
        save_path = output_dir / f"{safe_name}_spider.png"
        create_spider_chart(algorithms_data, title=f"{problem_name} - Algorithm Comparison", 
                          save_path=str(save_path))
    
    # Create comparison grid
    create_comparison_grid(spider_data, str(output_dir))
    
    # Create summary table
    create_summary_table(spider_data, str(output_dir / "discrete_summary.txt"))
    
    print("\nAll visualizations created successfully!")
    print(f"Charts saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
