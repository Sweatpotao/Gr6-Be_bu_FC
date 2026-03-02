"""
Visualization module for continuous algorithm comparison results.
Creates radar charts for comparing optimization algorithms (PSO, GA, DE, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import (
    CONTINUOUS_COLORS,
    calculate_angles,
    ensure_output_dir,
    save_or_show,
    normalize_metrics_minmax
)


def normalize_continuous_metrics(results_dict: Dict) -> Tuple[Dict, List[str]]:
    """
    Normalize continuous algorithm metrics to [0, 1] range.
    
    Metrics: mean_score, best_score, std_score, mean_time, mean_effort, success_rate
    All metrics: lower is better except success_rate (higher is better)
    
    Args:
        results_dict: Dict mapping algorithm name to performance stats
        
    Returns:
        Tuple of (normalized_data_dict, metric_labels)
    """
    # 6 criteria based on the experiment framework
    metrics_keys = ['mean_score', 'best_score', 'std_score', 'mean_time', 'mean_effort', 'success_rate']
    
    # Display labels for charts
    formatted_metrics = [
        'Avg Quality\n(mean_score)', 
        'Peak Quality\n(best_score)', 
        'Stability\n(std_score)', 
        'Speed\n(mean_time)', 
        'Comp. Cost\n(mean_effort)', 
        'Reliability\n(success_rate)'
    ]
    
    # Collect raw data
    raw_data = {key: [] for key in metrics_keys}
    algorithms = list(results_dict.keys())
    
    for algo in algorithms:
        summary = results_dict[algo]
        raw_data['mean_score'].append(summary.get('mean_score', 0) or 0)
        raw_data['best_score'].append(summary.get('best_score', 0) or 0)
        raw_data['std_score'].append(summary.get('std_score', 0) or 0)
        raw_data['mean_time'].append(summary.get('mean_time', 0) or 0)
        raw_data['mean_effort'].append(summary.get('mean_effort', 0) or 0)
        raw_data['success_rate'].append(summary.get('success_rate', 0) or 0)
    
    # Normalize each metric
    normalized_data = {}
    for key in metrics_keys:
        arr = np.array(raw_data[key])
        
        if np.max(arr) == np.min(arr):
            # All algorithms equal on this metric
            normalized_data[key] = np.ones_like(arr) if np.max(arr) > 0 else np.full_like(arr, 0.1)
        else:
            if key == 'success_rate':
                # Success Rate: higher is better
                norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            else:
                # Other metrics: lower is better (invert)
                norm = (np.max(arr) - arr) / (np.max(arr) - np.min(arr))
            
            # Scale to [0.1, 1.0] to avoid center piercing
            normalized_data[key] = 0.1 + (norm * 0.9)
    
    # Build per-algorithm normalized scores
    result = {}
    for i, algo in enumerate(algorithms):
        result[algo] = [normalized_data[key][i] for key in metrics_keys]
    
    return result, formatted_metrics


def plot_continuous_radar(problem_name: str, results_dict: Dict, 
                          save_path: Optional[str] = None,
                          title: Optional[str] = None) -> None:
    """
    Create radar chart for continuous algorithm comparison.
    
    Args:
        problem_name: Name of the problem
        results_dict: Dict mapping algorithm name to performance stats
        save_path: Path to save figure (if None, display only)
        title: Optional custom title
    """
    data_matrix, metrics = normalize_continuous_metrics(results_dict)
    algorithms = list(data_matrix.keys())
    
    if not algorithms:
        print("Warning: No algorithm data to plot")
        return

    num_vars = len(metrics)
    angles = calculate_angles(num_vars)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Setup axes
    plt.xticks(angles[:-1], metrics, color='black', size=10, weight='bold')
    ax.tick_params(axis='x', pad=20)  # Push labels away from edge
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)

    # Plot each algorithm
    for idx, algo in enumerate(algorithms):
        values = data_matrix[algo]
        values_closed = values + values[:1]  # Close the polygon
        
        color = CONTINUOUS_COLORS[idx % len(CONTINUOUS_COLORS)]
        ax.plot(angles, values_closed, linewidth=2, linestyle='solid', label=algo, color=color)
        ax.fill(angles, values_closed, alpha=0.15, color=color)

    # Title and legend
    chart_title = title or f"{problem_name.upper()} - Performance Comparison"
    plt.title(chart_title, size=16, weight='bold', y=1.15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    if save_path:
        ensure_output_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    [+] Saved radar chart: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_continuous_comparison_legacy(problem_name: str, results_dict: Dict, save_path: str):
    """
    Legacy radar chart plotting (from original plot_comparison.py).
    Maintained for backward compatibility.
    
    Args:
        problem_name: Name of the problem  
        results_dict: Dict with metrics per algorithm
        save_path: Path to save the figure
    """
    data_matrix, metrics = normalize_metrics_legacy(results_dict)
    algorithms = list(data_matrix.keys())
    
    if not algorithms:
        return

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Setup axes
    plt.xticks(angles[:-1], metrics, color='black', size=10, weight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "Best (1.0)"], color="grey", size=8)

    # Plot each algorithm
    for idx, algo in enumerate(algorithms):
        values = data_matrix[algo]
        values += values[:1]  # Close the polygon
        
        color = CONTINUOUS_COLORS[idx % len(CONTINUOUS_COLORS)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    plt.title(f"Algorithm Comparison: {problem_name.upper()}", size=14, weight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    ensure_output_dir(os.path.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def normalize_metrics_legacy(results_dict: Dict) -> Tuple[Dict, List[str]]:
    """
    Legacy normalization (5 metrics from original plot_comparison.py).
    """
    metrics = ['mean_score', 'best_score', 'std_score', 'mean_time', 'mean_effort']
    labels = ['Mean Score\n(Quality)', 'Best Score\n(Peak)', 'Std Score\n(Stability)', 
              'Mean Time\n(Speed)', 'Mean Effort\n(Evals)']
    
    # Find min/max for each metric
    min_v = {k: float('inf') for k in metrics}
    max_v = {k: float('-inf') for k in metrics}
    
    for algo, stats in results_dict.items():
        for k in metrics:
            val = stats.get(k, 0.0) if stats.get(k) is not None else 0.0
            if val < min_v[k]: 
                min_v[k] = val
            if val > max_v[k]: 
                max_v[k] = val

    # Normalize: invert so lower raw values = higher normalized (better)
    normalized = {}
    for algo, stats in results_dict.items():
        norm_scores = []
        for k in metrics:
            val = stats.get(k, 0.0) if stats.get(k) is not None else 0.0
            denom = max_v[k] - min_v[k]
            
            score = 1.0 if denom == 0 else (max_v[k] - val) / denom
            norm_scores.append(score)
            
        normalized[algo] = norm_scores
        
    return normalized, labels


# Export main function with clearer name
def plot_radar_chart(problem_name: str, results_dict: Dict, plots_dir: str):
    """
    Main entry point for plotting radar charts (compatible with run_all_experiments).
    
    Args:
        problem_name: Name of the problem
        results_dict: Dict of algorithm results
        plots_dir: Directory to save plots
    """
    plot_path = os.path.join(plots_dir, f"{problem_name}_radar.png")
    plot_continuous_radar(problem_name, results_dict, save_path=plot_path)
