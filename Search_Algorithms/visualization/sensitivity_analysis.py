"""
Parameter sensitivity analysis visualization for discrete optimization algorithms.

Provides tools for:
- 1D sensitivity analysis (line plots)
- 2D sensitivity analysis (heatmaps, contour plots)
- Parameter interaction analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from itertools import product
import json

from .base import ensure_output_dir, get_algorithm_color

# Try to import seaborn (optional dependency)
try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass  # Use default matplotlib style
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not installed. Using matplotlib default styles.")


def plot_parameter_sensitivity_1d(
    param_name: str,
    param_values: List[Any],
    results: Dict[str, List[float]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    ylabel: str = "Best Score",
    plot_type: str = "line"  # 'line' or 'bar'
):
    """
    Plot 1D parameter sensitivity analysis.
    Shows how algorithm performance changes with a single parameter.
    
    Args:
        param_name: Name of the parameter being analyzed
        param_values: List of parameter values tested
        results: Dict mapping algorithm name to list of scores for each parameter value
                Format: {'Algorithm': [score_for_val1, score_for_val2, ...]}
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        ylabel: Y-axis label
        plot_type: 'line' for line plot, 'bar' for bar chart
    
    Example:
        param_values = [20, 50, 100, 200]
        results = {
            'GA_TSP': [280, 260, 255, 258],
            'SA_TSP': [290, 270, 265, 268]
        }
        plot_parameter_sensitivity_1d('population_size', param_values, results)
    """
    if not results:
        warnings.warn("Empty results dictionary provided")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (algo, scores) in enumerate(results.items()):
        if len(scores) != len(param_values):
            warnings.warn(f"Length mismatch for {algo}: scores ({len(scores)}) vs param_values ({len(param_values)})")
            continue
        
        color = get_algorithm_color(algo, i)
        
        if plot_type == "line":
            ax.plot(param_values, scores, 'o-', linewidth=2, markersize=8,
                   label=algo, color=color)
        else:  # bar
            width = 0.8 / len(results)
            offset = (i - len(results)/2 + 0.5) * width
            x_pos = np.arange(len(param_values))
            ax.bar(x_pos + offset, scores, width, label=algo, color=color, alpha=0.7)
    
    if plot_type == "bar":
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels(param_values)
    
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title or f"Parameter Sensitivity: {param_name}", 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for best value
    if param_values:  # Check param_values is not empty
        for algo, scores in results.items():
            if scores:
                best_idx = np.argmin(scores)  # Assuming minimization
                if best_idx < len(param_values):
                    best_val = param_values[best_idx]
                    best_score = scores[best_idx]
                    ax.annotate(f'Best: {best_val}', 
                               xy=(best_val, best_score),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=8, alpha=0.7,
                               arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 1D sensitivity plot to {save_path}")
    
    return fig


def plot_parameter_sensitivity_2d(
    param1_name: str,
    param1_values: List[Any],
    param2_name: str,
    param2_values: List[Any],
    results_matrix: np.ndarray,
    algorithm_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    plot_type: str = "heatmap"  # 'heatmap' or 'contour'
):
    """
    Plot 2D parameter sensitivity analysis as heatmap or contour plot.
    Shows how performance varies with two parameters simultaneously.
    
    Args:
        param1_name: Name of first parameter
        param1_values: Values for first parameter
        param2_name: Name of second parameter
        param2_values: Values for second parameter
        results_matrix: 2D array of scores with shape (len(param1_values), len(param2_values))
        algorithm_name: Name of the algorithm
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        plot_type: 'heatmap' or 'contour'
    
    Example:
        pop_sizes = [20, 50, 100]
        mut_rates = [0.01, 0.05, 0.1]
        results = np.array([
            [280, 275, 290],  # pop=20
            [260, 255, 270],  # pop=50
            [265, 260, 275]   # pop=100
        ])
        plot_parameter_sensitivity_2d('pop_size', pop_sizes, 'mut_rate', mut_rates, 
                                     results, 'GA_TSP')
    """
    if results_matrix.shape != (len(param1_values), len(param2_values)):
        raise ValueError(f"Results matrix shape {results_matrix.shape} doesn't match "
                        f"parameters ({len(param1_values)}, {len(param2_values)})")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == "heatmap":
        # Create heatmap
        if HAS_SEABORN:
            sns.heatmap(results_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r',
                       xticklabels=param2_values, yticklabels=param1_values,
                       ax=ax, cbar_kws={'label': 'Score (lower is better)'})
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(results_matrix, cmap='RdYlGn_r', aspect='auto')
            # Add text annotations
            for i in range(len(param1_values)):
                for j in range(len(param2_values)):
                    text = ax.text(j, i, f'{results_matrix[i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)
            ax.set_xticks(range(len(param2_values)))
            ax.set_yticks(range(len(param1_values)))
            ax.set_xticklabels(param2_values)
            ax.set_yticklabels(param1_values)
            plt.colorbar(im, ax=ax, label='Score (lower is better)')
        ax.set_xlabel(param2_name.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel(param1_name.replace('_', ' ').title(), fontweight='bold')
        
    else:  # contour
        # Create contour plot
        X, Y = np.meshgrid(param2_values, param1_values)
        
        # Handle log scale for certain parameters
        if any(x in param1_name.lower() for x in ['size', 'count', 'temp']):
            ax.set_yscale('log')
        if any(x in param2_name.lower() for x in ['size', 'count', 'temp']):
            ax.set_xscale('log')
        
        contour = ax.contourf(X, Y, results_matrix, levels=20, cmap='RdYlGn_r')
        contour_lines = ax.contour(X, Y, results_matrix, levels=10, colors='black', linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        ax.set_xlabel(param2_name.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel(param1_name.replace('_', ' ').title(), fontweight='bold')
        plt.colorbar(contour, ax=ax, label='Score (lower is better)')
    
    ax.set_title(title or f"{algorithm_name}: 2D Parameter Sensitivity\n"
                        f"({param1_name} vs {param2_name})",
                fontsize=14, fontweight='bold', pad=20)
    
    # Highlight optimal point
    min_idx = np.unravel_index(np.argmin(results_matrix), results_matrix.shape)
    ax.plot(min_idx[1] + 0.5 if plot_type == "heatmap" else param2_values[min_idx[1]],
           min_idx[0] + 0.5 if plot_type == "heatmap" else param1_values[min_idx[0]],
           'r*', markersize=20, label=f'Optimal: ({param1_values[min_idx[0]]}, {param2_values[min_idx[1]]})')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D sensitivity plot to {save_path}")
    
    return fig


def plot_multi_algorithm_sensitivity(
    param_name: str,
    param_values: List[Any],
    results_by_algo: Dict[str, List[float]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7)
):
    """
    Compare parameter sensitivity across multiple algorithms in one plot.
    
    Args:
        param_name: Name of parameter
        param_values: List of parameter values
        results_by_algo: Dict mapping algorithm name to list of scores
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    if not results_by_algo:
        warnings.warn("Empty results dictionary provided")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (algo, scores) in enumerate(results_by_algo.items()):
        color = get_algorithm_color(algo, i)
        ax.plot(param_values, scores, 'o-', linewidth=2.5, markersize=10,
               label=algo, color=color, alpha=0.8)
    
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Score', fontsize=13, fontweight='bold')
    ax.set_title(title or f"Algorithm Comparison: Sensitivity to {param_name}",
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add shaded region for best performance
    all_scores = [score for scores in results_by_algo.values() for score in scores]
    if all_scores:
        best_score = min(all_scores)
        ax.axhspan(best_score * 0.98, best_score * 1.02, alpha=0.1, color='green',
                  label='Near-optimal region')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-algorithm sensitivity plot to {save_path}")
    
    return fig


def plot_parameter_importance(
    param_names: List[str],
    importance_scores: List[float],
    algorithm_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot parameter importance scores (e.g., from sensitivity analysis).
    
    Args:
        param_names: List of parameter names
        importance_scores: Importance score for each parameter
        algorithm_name: Name of algorithm
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_names = [param_names[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(param_names)))
    bars = ax.barh(sorted_names, sorted_scores, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(title or f"{algorithm_name}: Parameter Importance",
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, sorted_scores):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter importance plot to {save_path}")
    
    return fig


class ParameterSensitivityAnalyzer:
    """
    Class for running and visualizing parameter sensitivity analysis.
    
    Usage:
        analyzer = ParameterSensitivityAnalyzer(GA_TSP, problem)
        analyzer.analyze_1d('population_size', [20, 50, 100, 200])
        analyzer.analyze_2d('population_size', [20, 50, 100], 
                           'mutation_rate', [0.01, 0.05, 0.1])
        analyzer.plot_results(output_dir='sensitivity_results')
    """
    
    def __init__(self, algorithm_class: type, problem_instance: Any):
        """
        Initialize analyzer.
        
        Args:
            algorithm_class: Algorithm class to analyze
            problem_instance: Problem instance for testing
        """
        self.algorithm_class = algorithm_class
        self.problem = problem_instance
        self.results_1d = {}
        self.results_2d = {}
        
    def analyze_1d(self, param_name: str, param_values: List[Any],
                   fixed_params: Optional[Dict[str, Any]] = None,
                   n_runs: int = 5) -> Dict[str, List[float]]:
        """
        Run 1D sensitivity analysis.
        
        Args:
            param_name: Parameter to vary
            param_values: Values to test
            fixed_params: Fixed parameters for the algorithm
            n_runs: Number of runs per parameter value
            
        Returns:
            Dict with 'mean_scores', 'std_scores', 'best_scores'
        """
        fixed_params = fixed_params or {}
        mean_scores = []
        std_scores = []
        best_scores = []
        
        print(f"Running 1D sensitivity analysis for '{param_name}'...")
        
        for value in param_values:
            run_scores = []
            
            for run in range(n_runs):
                config = fixed_params.copy()
                config[param_name] = value
                
                try:
                    algo = self.algorithm_class(self.problem, config)
                    result = algo.run()
                    
                    if result.get('found'):
                        score = result.get('final_score', float('inf'))
                        # Use raw score (assumes minimization problem)
                        run_scores.append(score)
                except Exception as e:
                    warnings.warn(f"Run failed for {param_name}={value}: {e}")
                    run_scores.append(float('inf'))
            
            mean_scores.append(np.mean(run_scores) if run_scores else float('inf'))
            std_scores.append(np.std(run_scores) if run_scores else 0)
            best_scores.append(np.min(run_scores) if run_scores else float('inf'))
        
        self.results_1d[param_name] = {
            'values': param_values,
            'mean_scores': mean_scores,
            'std_scores': std_scores,
            'best_scores': best_scores
        }
        
        return self.results_1d[param_name]
    
    def analyze_2d(self, param1_name: str, param1_values: List[Any],
                   param2_name: str, param2_values: List[Any],
                   fixed_params: Optional[Dict[str, Any]] = None,
                   n_runs: int = 3) -> np.ndarray:
        """
        Run 2D sensitivity analysis.
        
        Args:
            param1_name: First parameter name
            param1_values: First parameter values
            param2_name: Second parameter name
            param2_values: Second parameter values
            fixed_params: Fixed parameters
            n_runs: Number of runs per combination
            
        Returns:
            2D array of mean scores
        """
        fixed_params = fixed_params or {}
        results_matrix = np.zeros((len(param1_values), len(param2_values)))
        
        print(f"Running 2D sensitivity analysis for '{param1_name}' vs '{param2_name}'...")
        
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                run_scores = []
                
                for run in range(n_runs):
                    config = fixed_params.copy()
                    config[param1_name] = val1
                    config[param2_name] = val2
                    
                    try:
                        algo = self.algorithm_class(self.problem, config)
                        result = algo.run()
                        
                        if result.get('found'):
                            score = result.get('final_score', float('inf'))
                            # Use raw score (assumes minimization problem)
                            run_scores.append(score)
                    except Exception as e:
                        warnings.warn(f"Run failed for {param1_name}={val1}, {param2_name}={val2}: {e}")
                        run_scores.append(float('inf'))
                
                results_matrix[i, j] = np.mean(run_scores) if run_scores else float('inf')
        
        self.results_2d[(param1_name, param2_name)] = {
            'param1_values': param1_values,
            'param2_values': param2_values,
            'results_matrix': results_matrix
        }
        
        return results_matrix
    
    def plot_results(self, output_dir: str = "sensitivity_analysis"):
        """
        Generate and save all sensitivity plots.
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = ensure_output_dir(output_dir)
        algo_name = self.algorithm_class.__name__
        
        # Plot 1D results
        for param_name, data in self.results_1d.items():
            plot_parameter_sensitivity_1d(
                param_name,
                data['values'],
                {algo_name: data['best_scores']},
                title=f"{algo_name}: Sensitivity to {param_name}",
                save_path=str(output_path / f"{algo_name}_{param_name}_1d.png")
            )
            plt.close()
        
        # Plot 2D results
        for (p1, p2), data in self.results_2d.items():
            plot_parameter_sensitivity_2d(
                p1, data['param1_values'],
                p2, data['param2_values'],
                data['results_matrix'],
                algo_name,
                save_path=str(output_path / f"{algo_name}_{p1}_vs_{p2}_2d.png")
            )
            plt.close()
        
        # Helper function to convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        # Save results to JSON
        results_json = {
            'algorithm': algo_name,
            '1d_analysis': convert_to_native(self.results_1d),
            '2d_analysis': {
                f"{k[0]}_vs_{k[1]}": {
                    'param1_values': convert_to_native(v['param1_values']),
                    'param2_values': convert_to_native(v['param2_values']),
                    'results_matrix': convert_to_native(v['results_matrix'])
                }
                for k, v in self.results_2d.items()
            }
        }
        
        with open(output_path / f"{algo_name}_sensitivity_results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Sensitivity analysis results saved to {output_path}")


def create_sensitivity_report(
    algorithm_configs: Dict[str, Dict[str, List[Any]]],
    problem_instances: Dict[str, Any],
    output_dir: str = "sensitivity_reports"
):
    """
    Create comprehensive sensitivity analysis report for multiple algorithms.
    
    Args:
        algorithm_configs: Dict mapping algorithm name to config with param ranges
                          Format: {'GA_TSP': {'class': GA_TSP, 'params': {...}}}
        problem_instances: Dict mapping problem name to problem instance
        output_dir: Directory to save reports
    """
    output_path = ensure_output_dir(output_dir)
    
    for algo_name, config in algorithm_configs.items():
        algo_class = config['class']
        param_ranges = config['params']
        problem_name = config.get('problem', list(problem_instances.keys())[0])
        problem = problem_instances[problem_name]
        
        print(f"\n{'='*60}")
        print(f"Analyzing {algo_name} on {problem_name}")
        print(f"{'='*60}")
        
        analyzer = ParameterSensitivityAnalyzer(algo_class, problem)
        
        # Run 1D analysis for each parameter
        for param_name, param_values in param_ranges.items():
            if isinstance(param_values, list):
                analyzer.analyze_1d(param_name, param_values, n_runs=5)
        
        # Run 2D analysis for parameter pairs
        param_names = list(param_ranges.keys())
        for i in range(len(param_names)):
            for j in range(i+1, len(param_names)):
                p1, p2 = param_names[i], param_names[j]
                if isinstance(param_ranges[p1], list) and isinstance(param_ranges[p2], list):
                    analyzer.analyze_2d(p1, param_ranges[p1][:4],  # Limit to 4 values
                                       p2, param_ranges[p2][:4], n_runs=3)
        
        # Generate plots
        algo_output = output_path / algo_name
        analyzer.plot_results(algo_output)
    
    print(f"\nAll sensitivity reports saved to {output_path}")
