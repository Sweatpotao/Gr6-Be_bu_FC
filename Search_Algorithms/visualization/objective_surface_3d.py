"""
3D visualization of objective function surfaces for discrete optimization problems.

Provides tools for:
- 3D surface plots of objective functions
- Fitness landscape visualization
- Search trajectory visualization in 3D space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from itertools import combinations

from .base import ensure_output_dir, get_algorithm_color


def plot_tsp_objective_surface(
    distance_matrix: np.ndarray,
    sample_permutations: Optional[List[Tuple]] = None,
    title: str = "TSP Objective Function Landscape",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 9)
):
    """
    Create 3D visualization of TSP objective function.
    
    Since TSP is discrete with factorial search space, we sample random permutations
    and project them onto 2D using PCA or t-SNE for visualization.
    
    Args:
        distance_matrix: Distance matrix for TSP
        sample_permutations: Optional list of (permutation, cost) tuples to plot
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    n_cities = len(distance_matrix)
    
    # Generate sample permutations if not provided
    if sample_permutations is None:
        sample_permutations = []
        np.random.seed(42)
        
        # Generate random permutations and compute costs
        for _ in range(500):
            perm = tuple(np.random.permutation(n_cities))
            cost = compute_tsp_cost(perm, distance_matrix)
            sample_permutations.append((perm, cost))
    
    # Extract permutations and costs
    perms = [p[0] for p in sample_permutations]
    costs = np.array([p[1] for p in sample_permutations])
    
    # Project permutations to 2D using simple dimensionality reduction
    # Use first two cities' positions as x, y coordinates
    coords = np.array([[p[0], p[1] if len(p) > 1 else 0] for p in perms])
    
    # Add small jitter for better visualization
    coords = coords + np.random.normal(0, 0.1, coords.shape)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with color based on cost
    scatter = ax.scatter(coords[:, 0], coords[:, 1], costs,
                        c=costs, cmap='viridis_r', s=50, alpha=0.6,
                        edgecolors='black', linewidth=0.5)
    
    # Add surface interpolation
    if len(coords) > 10:
        try:
            from scipy.interpolate import griddata
            xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 50)
            yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((coords[:, 0], coords[:, 1]), costs, (xi, yi), method='cubic')
            
            # Plot surface
            ax.plot_surface(xi, yi, zi, alpha=0.3, cmap='viridis_r', 
                          rstride=1, cstride=1, antialiased=True)
        except ImportError:
            warnings.warn("scipy not available, skipping surface interpolation")
    
    ax.set_xlabel('Dimension 1 (Permutation Index)', fontweight='bold')
    ax.set_ylabel('Dimension 2 (Permutation Index)', fontweight='bold')
    ax.set_zlabel('Tour Cost', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Cost (lower is better)', fontweight='bold')
    
    # Mark best solution
    best_idx = np.argmin(costs)
    ax.scatter(coords[best_idx, 0], coords[best_idx, 1], costs[best_idx],
              c='red', s=200, marker='*', edgecolors='black', linewidth=2,
              label=f'Best: {costs[best_idx]:.1f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved TSP 3D surface to {save_path}")
    
    return fig


def compute_tsp_cost(permutation: Tuple, distance_matrix: np.ndarray) -> float:
    """Compute TSP tour cost for a given permutation."""
    cost = 0
    n = len(permutation)
    for i in range(n):
        cost += distance_matrix[permutation[i]][permutation[(i+1) % n]]
    return cost


def plot_knapsack_objective_surface(
    weights: List[float],
    values: List[float],
    capacity: float,
    sample_points: int = 1000,
    title: str = "Knapsack Objective Function Landscape",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 9)
):
    """
    Create 3D visualization of Knapsack objective function.
    
    Visualizes the relationship between:
    - X: Total weight of selected items
    - Y: Number of selected items
    - Z: Total value (objective function)
    
    Args:
        weights: Item weights
        values: Item values
        capacity: Knapsack capacity
        sample_points: Number of random samples to generate
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    n_items = len(weights)
    
    # Generate random valid solutions
    np.random.seed(42)
    samples = []
    
    for _ in range(sample_points):
        # Random selection
        selection = np.random.choice([0, 1], size=n_items)
        total_weight = np.sum(selection * weights)
        total_value = np.sum(selection * values)
        n_selected = np.sum(selection)
        
        # Only include valid solutions (weight <= capacity)
        if total_weight <= capacity:
            samples.append((total_weight, n_selected, total_value))
    
    if not samples:
        warnings.warn("No valid samples generated")
        return None
    
    samples = np.array(samples)
    weights_selected = samples[:, 0]
    n_selected = samples[:, 1]
    total_values = samples[:, 2]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(weights_selected, n_selected, total_values,
                        c=total_values, cmap='RdYlGn', s=30, alpha=0.6,
                        edgecolors='black', linewidth=0.3)
    
    # Add capacity plane
    weight_range = np.linspace(0, capacity, 20)
    n_range = np.linspace(0, n_items, 20)
    W, N = np.meshgrid(weight_range, n_range)
    
    # Approximate surface
    try:
        from scipy.interpolate import griddata
        Z = griddata((weights_selected, n_selected), total_values, (W, N), 
                    method='linear', fill_value=0)
        ax.plot_surface(W, N, Z, alpha=0.3, cmap='RdYlGn', 
                       rstride=1, cstride=1, antialiased=True)
    except ImportError:
        pass
    
    # Mark capacity constraint
    ax.plot([capacity, capacity], [0, n_items], 
           [total_values.min(), total_values.max()],
           'r--', linewidth=3, label=f'Capacity: {capacity}')
    
    ax.set_xlabel('Total Weight', fontweight='bold')
    ax.set_ylabel('Number of Items', fontweight='bold')
    ax.set_zlabel('Total Value', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value (higher is better)', fontweight='bold')
    
    # Mark best solution
    best_idx = np.argmax(total_values)
    ax.scatter(weights_selected[best_idx], n_selected[best_idx], total_values[best_idx],
              c='blue', s=200, marker='*', edgecolors='black', linewidth=2,
              label=f'Best Value: {total_values[best_idx]:.1f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Knapsack 3D surface to {save_path}")
    
    return fig


def plot_nqueens_objective_surface(
    n: int = 8,
    sample_points: int = 2000,
    title: str = "N-Queens Objective Function Landscape",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 9)
):
    """
    Create 3D visualization of N-Queens objective function (number of conflicts).
    
    Visualizes:
    - X, Y: Position coordinates
    - Z: Number of conflicts (objective to minimize)
    
    Args:
        n: Board size (N)
        sample_points: Number of random board configurations to sample
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    np.random.seed(42)
    
    # Generate random solutions (permutations represent queen positions)
    samples = []
    for _ in range(sample_points):
        board = tuple(np.random.permutation(n))
        conflicts = count_nqueens_conflicts(board)
        # Use first two queen positions as coordinates
        samples.append((board[0], board[1], conflicts))
    
    samples = np.array(samples)
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter with color based on conflicts
    scatter = ax.scatter(x, y, z, c=z, cmap='RdYlGn_r', s=40, alpha=0.6,
                        edgecolors='black', linewidth=0.3)
    
    # Add surface
    try:
        from scipy.interpolate import griddata
        xi = np.linspace(0, n-1, 30)
        yi = np.linspace(0, n-1, 30)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='linear', fill_value=np.max(z))
        ax.plot_surface(xi, yi, zi, alpha=0.3, cmap='RdYlGn_r',
                       rstride=1, cstride=1, antialiased=True)
    except ImportError:
        pass
    
    ax.set_xlabel('Queen 1 Position', fontweight='bold')
    ax.set_ylabel('Queen 2 Position', fontweight='bold')
    ax.set_zlabel('Number of Conflicts', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Conflicts (lower is better)', fontweight='bold')
    
    # Mark zero conflict solutions
    zero_conflict_mask = z == 0
    if np.any(zero_conflict_mask):
        ax.scatter(x[zero_conflict_mask], y[zero_conflict_mask], z[zero_conflict_mask],
                  c='green', s=200, marker='*', edgecolors='black', linewidth=2,
                  label=f'Valid Solutions: {np.sum(zero_conflict_mask)}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved N-Queens 3D surface to {save_path}")
    
    return fig


def count_nqueens_conflicts(board: Tuple[int, ...]) -> int:
    """Count number of attacking queen pairs."""
    n = len(board)
    conflicts = 0
    for i in range(n):
        for j in range(i+1, n):
            # Same diagonal
            if abs(board[i] - board[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def plot_search_trajectory_3d(
    trajectory: List[Tuple],
    fitness_values: List[float],
    algorithm_name: str,
    title: str = "Search Trajectory in 3D",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 9),
    show_surface: bool = True
):
    """
    Visualize algorithm search trajectory in 3D space.
    
    Args:
        trajectory: List of (x, y) coordinates representing search positions
        fitness_values: Fitness value at each position
        algorithm_name: Name of the algorithm
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        show_surface: Whether to show interpolated surface
    """
    if len(trajectory) != len(fitness_values):
        raise ValueError("Trajectory and fitness_values must have same length")
    
    trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = np.array(fitness_values)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by iteration
    colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory)))
    
    # Plot trajectory line
    for i in range(len(trajectory) - 1):
        ax.plot3D(x[i:i+2], y[i:i+2], z[i:i+2], 
                 color=colors[i], linewidth=2, alpha=0.7)
    
    # Plot points with color gradient
    scatter = ax.scatter(x, y, z, c=range(len(trajectory)), 
                        cmap='plasma', s=60, edgecolors='black', linewidth=0.5)
    
    # Mark start and end
    ax.scatter(x[0], y[0], z[0], c='green', s=200, marker='o', 
              edgecolors='black', linewidth=2, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=200, marker='*', 
              edgecolors='black', linewidth=2, label='End')
    
    # Add surface if requested
    if show_surface and len(trajectory) > 10:
        try:
            from scipy.interpolate import griddata
            xi = np.linspace(x.min(), x.max(), 30)
            yi = np.linspace(y.min(), y.max(), 30)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x, y), z, (xi, yi), method='linear', fill_value=np.mean(z))
            ax.plot_surface(xi, yi, zi, alpha=0.2, cmap='viridis', 
                          rstride=1, cstride=1, antialiased=True)
        except ImportError:
            pass
    
    ax.set_xlabel('X Dimension', fontweight='bold')
    ax.set_ylabel('Y Dimension', fontweight='bold')
    ax.set_zlabel('Fitness', fontweight='bold')
    ax.set_title(f"{title}\n{algorithm_name}", fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Iteration', fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D trajectory to {save_path}")
    
    return fig


def plot_multiple_trajectories_3d(
    trajectories: Dict[str, List[Tuple]],
    fitness_histories: Dict[str, List[float]],
    title: str = "Algorithm Comparison: Search Trajectories",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Compare search trajectories of multiple algorithms in 3D.
    
    Args:
        trajectories: Dict mapping algorithm name to list of positions
        fitness_histories: Dict mapping algorithm name to list of fitness values
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (algo_name, traj) in enumerate(trajectories.items()):
        if algo_name not in fitness_histories:
            continue
        
        traj = np.array(traj)
        fitness = np.array(fitness_histories[algo_name])
        
        if len(traj) != len(fitness):
            continue
        
        color = get_algorithm_color(algo_name, i)
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], fitness, 
               color=color, linewidth=2, label=algo_name, alpha=0.8)
        
        # Mark final position
        ax.scatter(traj[-1, 0], traj[-1, 1], fitness[-1],
                  c=color, s=150, marker='*', edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('X Dimension', fontweight='bold')
    ax.set_ylabel('Y Dimension', fontweight='bold')
    ax.set_zlabel('Fitness', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multiple trajectories plot to {save_path}")
    
    return fig


def create_3d_visualization_suite(
    problem_type: str,
    problem_data: Dict[str, Any],
    output_dir: str = "3d_visualizations"
):
    """
    Create comprehensive 3D visualizations for a problem type.
    
    Args:
        problem_type: 'tsp', 'knapsack', or 'nqueens'
        problem_data: Problem-specific data
        output_dir: Directory to save visualizations
    """
    output_path = ensure_output_dir(output_dir)
    
    print(f"\nCreating 3D visualizations for {problem_type}...")
    
    if problem_type.lower() == 'tsp':
        distance_matrix = problem_data.get('distance_matrix')
        if distance_matrix is not None:
            plot_tsp_objective_surface(
                np.array(distance_matrix),
                title="TSP: Objective Function Landscape",
                save_path=str(output_path / "tsp_3d_surface.png")
            )
            plt.close()
    
    elif problem_type.lower() == 'knapsack':
        weights = problem_data.get('weights', [])
        values = problem_data.get('values', [])
        capacity = problem_data.get('capacity', 0)
        
        if weights and values and capacity:
            plot_knapsack_objective_surface(
                weights, values, capacity,
                title="Knapsack: Objective Function Landscape",
                save_path=str(output_path / "knapsack_3d_surface.png")
            )
            plt.close()
    
    elif problem_type.lower() == 'nqueens':
        n = problem_data.get('n', 8)
        plot_nqueens_objective_surface(
            n=n,
            title=f"N-Queens ({n}x{n}): Objective Function Landscape",
            save_path=str(output_path / f"nqueens_{n}_3d_surface.png")
        )
        plt.close()
    
    print(f"3D visualizations saved to {output_path}")
    return output_path
