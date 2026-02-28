"""
Base utilities for visualization modules.
Shared helpers for discrete and continuous algorithm visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Color schemes
DISCRETE_COLOR_MAP = {
    # Classical algorithms
    'BFS': '#1f77b4',                    # Blue
    'DFS': '#4a90d9',                    # Light Blue
    'UCS': '#2ca02c',                    # Green
    'Greedy': '#ff7f0e',                 # Orange
    'A*': '#d62728',                     # Red
    'AStar': '#d62728',                  # Alias for A*
    # Metaheuristic - Local Search
    'HillClimbingTSP': '#9467bd',        # Purple
    'SimulatedAnnealingTSP': '#8c564b',  # Brown
    # Metaheuristic - Evolution
    'GA_TSP': '#e377c2',                 # Pink
    # Metaheuristic - Swarm
    'ABC_Knapsack': '#7f7f7f',           # Gray
    # Metaheuristic - Human
    'TLBO_Knapsack': '#bcbd22',          # Yellow-green
}

CONTINUOUS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def setup_polar_axes(ax, categories: List[str], ylim: Tuple[float, float] = (0, 1),
                     yticks: Optional[List[float]] = None, 
                     yticklabels: Optional[List[str]] = None):
    """
    Setup common polar axes configuration for radar charts.
    
    Args:
        ax: Matplotlib axes object (polar projection)
        categories: List of category labels
        ylim: Tuple of (min, max) for y-axis
        yticks: List of tick positions
        yticklabels: List of tick labels
    """
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Set x-ticks (categories)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    
    # Set y-limits
    ax.set_ylim(ylim)
    
    # Set y-ticks
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, color="grey", size=8)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return angles


def get_algorithm_color(algorithm_name: str, index: int = 0, 
                        color_map: Optional[Dict[str, str]] = None) -> str:
    """
    Get color for an algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        index: Fallback index for color cycling
        color_map: Optional custom color map
        
    Returns:
        Color hex code
    """
    if color_map is None:
        color_map = DISCRETE_COLOR_MAP
    
    # Try exact match first
    if algorithm_name in color_map:
        return color_map[algorithm_name]
    
    # Try case-insensitive match
    for key, color in color_map.items():
        if key.lower() == algorithm_name.lower():
            return color
    
    # Fallback to Set2 colormap
    return plt.cm.Set2(index / 8)


def normalize_metrics_minmax(values: List[float], invert: bool = True) -> List[float]:
    """
    Normalize values to [0, 1] range using min-max normalization.
    
    Args:
        values: List of raw values
        invert: If True, invert so lower raw values = higher normalized (better)
        
    Returns:
        List of normalized values in [0, 1]
    """
    if not values or len(values) == 0:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return [1.0] * len(values) if max_val > 0 else [0.5] * len(values)
    
    if invert:
        # Lower is better: (Max - Val) / (Max - Min)
        return [(max_val - v) / (max_val - min_val) for v in values]
    else:
        # Higher is better: (Val - Min) / (Max - Min)
        return [(v - min_val) / (max_val - min_val) for v in values]


def calculate_angles(num_vars: int) -> List[float]:
    """Calculate angles for radar chart axes."""
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    return angles


def ensure_output_dir(path: str) -> Path:
    """Ensure output directory exists."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_or_show(fig, save_path: Optional[str] = None, dpi: int = 300):
    """
    Save figure to file or display it.
    
    Args:
        fig: Matplotlib figure
        save_path: Path to save (if None, show instead)
        dpi: DPI for saved figure
    """
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()
    plt.close()
