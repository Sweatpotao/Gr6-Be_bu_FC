"""
Performance comparison charts for discrete optimization algorithms.

Provides comprehensive visualization for:
- Solution quality comparison (bar charts)
- Stability analysis (box plots)
- Convergence behavior (line plots)
- Multi-criteria comparison (grouped bar charts)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .base import ensure_output_dir, get_algorithm_color, DISCRETE_COLOR_MAP

# Try to import seaborn (optional dependency)
try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set default style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not installed. Using matplotlib default styles.")


def plot_performance_bar(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: List[str] = None,
    title: str = "Algorithm Performance Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    normalize: bool = True
):
    """
    Create grouped bar chart comparing algorithm performance across multiple metrics.
    
    Args:
        results_dict: Dict mapping algorithm name to performance metrics
                     Format: {'Algorithm': {'mean_score': x, 'best_score': y, ...}}
        metrics: List of metric keys to compare. If None, uses all available metrics.
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size (width, height)
        normalize: Whether to normalize metrics to [0, 1] for comparison
    
    Example:
        results = {
            'GA_TSP': {'mean_score': 255, 'best_score': 250, 'mean_time': 0.5},
            'SA_TSP': {'mean_score': 260, 'best_score': 255, 'mean_time': 0.3}
        }
        plot_performance_bar(results, metrics=['mean_score', 'best_score'])
    """
    if not results_dict:
        warnings.warn("Empty results dictionary provided")
        return
    
    algorithms = list(results_dict.keys())
    
    # Auto-detect metrics if not specified
    if metrics is None:
        sample_metrics = set()
        for algo_data in results_dict.values():
            sample_metrics.update(algo_data.keys())
        # Filter to numeric metrics
        metrics = []
        for m in sample_metrics:
            try:
                val = next((results_dict[a].get(m) for a in algorithms if results_dict[a].get(m) is not None), None)
                if val is not None and isinstance(val, (int, float)):
                    metrics.append(m)
            except:
                pass
    
    if not metrics:
        warnings.warn("No valid metrics found in results")
        return
    
    n_metrics = len(metrics)
    n_algorithms = len(algorithms)
    
    # Prepare data matrix
    data_matrix = np.zeros((n_algorithms, n_metrics))
    for i, algo in enumerate(algorithms):
        for j, metric in enumerate(metrics):
            value = results_dict[algo].get(metric, 0)
            # Handle negative scores (for maximization problems like Knapsack)
            data_matrix[i, j] = abs(value) if value is not None else 0
    
    # Normalize if requested (for fair comparison across different scales)
    if normalize:
        for j in range(n_metrics):
            col = data_matrix[:, j]
            col_min, col_max = np.min(col), np.max(col)
            if col_max > col_min:
                # For metrics where lower is better (time, std), invert
                metric_name = metrics[j].lower()
                if any(x in metric_name for x in ['time', 'std', 'effort']):
                    data_matrix[:, j] = 1 - (col - col_min) / (col_max - col_min)
                else:
                    data_matrix[:, j] = (col - col_min) / (col_max - col_min)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(n_metrics)
    width = 0.8 / n_algorithms
    
    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        offset = (i - n_algorithms/2 + 0.5) * width
        color = get_algorithm_color(algo, i)
        bars = ax.bar(x + offset, data_matrix[i], width, 
                     label=algo, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, data_matrix[i]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=7, rotation=0)
    
    # Configure axes
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Score' if normalize else 'Raw Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.2 if normalize else None)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance bar chart to {save_path}")
    
    return fig


def plot_performance_box(
    results_dict: Dict[str, List[float]],
    title: str = "Solution Quality Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    ylabel: str = "Score"
):
    """
    Create box plot showing solution distribution across multiple runs.
    Useful for analyzing stability and consistency.
    
    Args:
        results_dict: Dict mapping algorithm name to list of scores from multiple runs
                     Format: {'Algorithm': [score1, score2, score3, ...]}
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        ylabel: Label for Y-axis
    
    Example:
        results = {
            'GA_TSP': [255, 260, 252, 258, 255],
            'SA_TSP': [260, 265, 258, 262, 260]
        }
        plot_performance_box(results)
    """
    if not results_dict:
        warnings.warn("Empty results dictionary provided")
        return
    
    algorithms = list(results_dict.keys())
    data = [results_dict[algo] for algo in algorithms]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True, notch=True)
    
    # Color the boxes
    for i, patch in enumerate(bp['boxes']):
        color = get_algorithm_color(algorithms[i], i)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the plot
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x labels if needed
    if len(algorithms) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance box plot to {save_path}")
    
    return fig


def plot_convergence_comparison(
    histories: Dict[str, List[float]],
    title: str = "Convergence Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    xlabel: str = "Iteration",
    ylabel: str = "Best Score",
    log_scale: bool = False
):
    """
    Plot convergence curves for multiple algorithms.
    Shows how algorithms improve over iterations.
    
    Args:
        histories: Dict mapping algorithm name to list of best scores over iterations
                  Format: {'Algorithm': [best_score_iter1, best_score_iter2, ...]}
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Whether to use log scale for Y-axis
    
    Example:
        histories = {
            'GA_TSP': [500, 480, 450, 400, 350, 300, 280, 260, 255],
            'SA_TSP': [500, 490, 470, 440, 410, 380, 350, 320, 290, 270, 260]
        }
        plot_convergence_comparison(histories)
    """
    if not histories:
        warnings.warn("Empty histories dictionary provided")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (algo, history) in enumerate(histories.items()):
        if not history:
            continue
        color = get_algorithm_color(algo, i)
        iterations = range(len(history))
        ax.plot(iterations, history, label=algo, color=color, 
               linewidth=2, marker='o', markersize=4, markevery=max(1, len(history)//20))
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence comparison to {save_path}")
    
    return fig


def plot_stability_analysis(
    results_dict: Dict[str, Dict[str, Any]],
    title: str = "Algorithm Stability Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Create comprehensive stability analysis chart showing:
    - Mean score with error bars (std deviation)
    - Success rate
    - Best score achieved
    
    Args:
        results_dict: Dict mapping algorithm name to performance stats
                     Required keys: 'mean_score', 'std_score', 'best_score', 'success_rate'
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    if not results_dict:
        warnings.warn("Empty results dictionary provided")
        return
    
    algorithms = list(results_dict.keys())
    n_algos = len(algorithms)
    
    # Extract data
    means = [results_dict[a].get('mean_score', 0) for a in algorithms]
    stds = [results_dict[a].get('std_score', 0) for a in algorithms]
    bests = [results_dict[a].get('best_score', 0) for a in algorithms]
    success_rates = [results_dict[a].get('success_rate', 0) * 100 for a in algorithms]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Mean with error bars
    ax1 = axes[0, 0]
    colors = [get_algorithm_color(a, i) for i, a in enumerate(algorithms)]
    x_pos = np.arange(n_algos)
    ax1.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_ylabel('Mean Score', fontweight='bold')
    ax1.set_title('Mean Score ± Std Dev', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Best scores
    ax2 = axes[0, 1]
    ax2.bar(x_pos, bests, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_ylabel('Best Score', fontweight='bold')
    ax2.set_title('Best Score Achieved', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Success rate
    ax3 = axes[1, 0]
    ax3.bar(x_pos, success_rates, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.set_ylabel('Success Rate (%)', fontweight='bold')
    ax3.set_title('Success Rate', fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Coefficient of Variation (CV = std/mean)
    ax4 = axes[1, 1]
    cv = [s/m if m != 0 else 0 for s, m in zip(stds, means)]
    ax4.bar(x_pos, cv, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(algorithms, rotation=45, ha='right')
    ax4.set_ylabel('Coefficient of Variation', fontweight='bold')
    ax4.set_title('Stability (CV = std/mean)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved stability analysis to {save_path}")
    
    return fig


def plot_computation_efficiency(
    results_dict: Dict[str, Dict[str, Any]],
    title: str = "Computational Efficiency Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot analyzing computational efficiency: time vs quality trade-off.
    
    Args:
        results_dict: Dict with 'mean_time', 'mean_score' keys
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    if not results_dict:
        warnings.warn("Empty results dictionary provided")
        return
    
    algorithms = list(results_dict.keys())
    
    # Extract data
    times = [results_dict[a].get('mean_time', 0) for a in algorithms]
    scores = [abs(results_dict[a].get('mean_score', 0)) for a in algorithms]
    efforts = [results_dict[a].get('mean_effort', 0) for a in algorithms]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Time vs Quality scatter
    ax1 = axes[0]
    for i, algo in enumerate(algorithms):
        color = get_algorithm_color(algo, i)
        ax1.scatter(times[i], scores[i], s=200, c=color, alpha=0.7, 
                   edgecolors='black', linewidth=1.5, label=algo)
        ax1.annotate(algo, (times[i], scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Mean Time (s)', fontweight='bold')
    ax1.set_ylabel('Mean Score (lower is better)', fontweight='bold')
    ax1.set_title('Time vs Solution Quality', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Effort comparison
    ax2 = axes[1]
    colors = [get_algorithm_color(a, i) for i, a in enumerate(algorithms)]
    bars = ax2.barh(algorithms, efforts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mean Effort (evaluations)', fontweight='bold')
    ax2.set_title('Computational Effort', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, efforts):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(val)}', ha='left', va='center', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved efficiency analysis to {save_path}")
    
    return fig


def create_performance_dashboard(
    results_dict: Dict[str, Dict[str, Any]],
    problem_name: str,
    output_dir: str = "comparison_charts",
    convergence_histories: Optional[Dict[str, List[float]]] = None
):
    """
    Create comprehensive performance dashboard with multiple charts.
    
    Args:
        results_dict: Performance metrics for each algorithm
        problem_name: Name of the problem
        output_dir: Directory to save charts
        convergence_histories: Optional convergence histories
    """
    output_path = ensure_output_dir(output_dir)
    
    print(f"\nCreating performance dashboard for {problem_name}...")
    
    # 1. Multi-metric bar chart
    metrics = ['mean_score', 'best_score', 'std_score', 'mean_time']
    plot_performance_bar(
        results_dict,
        metrics=metrics,
        title=f"{problem_name}: Multi-Metric Comparison",
        save_path=str(output_path / f"{problem_name.lower()}_performance_bar.png")
    )
    plt.close()
    
    # 2. Stability analysis
    plot_stability_analysis(
        results_dict,
        title=f"{problem_name}: Stability Analysis",
        save_path=str(output_path / f"{problem_name.lower()}_stability.png")
    )
    plt.close()
    
    # 3. Computational efficiency
    plot_computation_efficiency(
        results_dict,
        title=f"{problem_name}: Efficiency Analysis",
        save_path=str(output_path / f"{problem_name.lower()}_efficiency.png")
    )
    plt.close()
    
    # 4. Convergence comparison (if histories provided)
    if convergence_histories:
        plot_convergence_comparison(
            convergence_histories,
            title=f"{problem_name}: Convergence Comparison",
            save_path=str(output_path / f"{problem_name.lower()}_convergence.png")
        )
        plt.close()
    
    print(f"Dashboard saved to {output_path}")
    return output_path


# Utility function to parse result text files
def parse_results_file(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse result text file and return structured data.
    
    Args:
        filepath: Path to results text file
        
    Returns:
        Dict mapping algorithm name to performance metrics
    """
    results = {}
    current_algo = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Algorithm'):
                current_algo = line.split(':')[1].strip()
                results[current_algo] = {}
            elif current_algo and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Try to convert to number
                try:
                    if '.' in value or 'e' in value.lower():
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Remove % sign if present
                    if value.endswith('%'):
                        try:
                            value = float(value[:-1]) / 100
                        except:
                            pass
                
                results[current_algo][key] = value
    
    return results
