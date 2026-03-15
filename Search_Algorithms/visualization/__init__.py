"""
Visualization module for Search_Algorithms project.

Provides comprehensive visualization for:
- Discrete algorithms (BFS, DFS, UCS, Greedy, A*)
- Metaheuristic algorithms (GA, SA, ABC, TLBO, ACO)
- Continuous algorithms (PSO, GA, DE, etc.)
- Performance comparison charts
- Parameter sensitivity analysis
- 3D objective surface visualization
"""

from .discrete_comparison import (
    create_spider_chart,
    create_comparison_grid,
    create_summary_table,
)

from .continuous_comparison import (
    plot_continuous_radar,
    plot_radar_chart,
    normalize_continuous_metrics,
)

from .performance_charts import (
    plot_performance_bar,
    plot_performance_box,
    plot_convergence_comparison,
    plot_stability_analysis,
    plot_computation_efficiency,
    create_performance_dashboard,
    parse_results_file,
)

from .sensitivity_analysis import (
    plot_parameter_sensitivity_1d,
    plot_parameter_sensitivity_2d,
    plot_multi_algorithm_sensitivity,
    plot_parameter_importance,
    ParameterSensitivityAnalyzer,
    create_sensitivity_report,
)

from .objective_surface_3d import (
    plot_tsp_objective_surface,
    plot_knapsack_objective_surface,
    plot_nqueens_objective_surface,
    plot_graph_coloring_objective_surface,
    plot_search_trajectory_3d,
    plot_multiple_trajectories_3d,
    create_3d_visualization_suite,
)

from .generate_charts_from_results import (
    parse_discrete_results,
    is_continuous_problem,
    generate_problem_visualizations,
    generate_3d_surface_from_problem,
    generate_summary_report,
)

from .base import (
    DISCRETE_COLOR_MAP,
    CONTINUOUS_COLORS,
    setup_polar_axes,
    get_algorithm_color,
    normalize_metrics_minmax,
    ensure_output_dir,
)

__all__ = [
    # Discrete visualization (existing)
    'create_spider_chart',
    'create_comparison_grid',
    'create_summary_table',
    # Continuous visualization (existing)
    'plot_continuous_radar',
    'plot_radar_chart',
    'normalize_continuous_metrics',
    # Performance comparison (new)
    'plot_performance_bar',
    'plot_performance_box',
    'plot_convergence_comparison',
    'plot_stability_analysis',
    'plot_computation_efficiency',
    'create_performance_dashboard',
    'parse_results_file',
    # Sensitivity analysis (new)
    'plot_parameter_sensitivity_1d',
    'plot_parameter_sensitivity_2d',
    'plot_multi_algorithm_sensitivity',
    'plot_parameter_importance',
    'ParameterSensitivityAnalyzer',
    'create_sensitivity_report',
    # 3D visualization (new)
    'plot_tsp_objective_surface',
    'plot_knapsack_objective_surface',
    'plot_nqueens_objective_surface',
    'plot_graph_coloring_objective_surface',
    'plot_search_trajectory_3d',
    'plot_multiple_trajectories_3d',
    'create_3d_visualization_suite',
    # Chart generation from results
    'parse_discrete_results',
    'is_continuous_problem',
    'generate_problem_visualizations',
    'generate_3d_surface_from_problem',
    'generate_summary_report',
    # Utilities
    'DISCRETE_COLOR_MAP',
    'CONTINUOUS_COLORS',
    'setup_polar_axes',
    'get_algorithm_color',
    'normalize_metrics_minmax',
    'ensure_output_dir',
]

__version__ = '2.0.0'  # Added performance, sensitivity, and 3D visualization modules


# Convenience function to create all visualizations at once
def create_all_visualizations(
    results_dict: dict,
    problem_name: str,
    output_dir: str = "visualization_output",
    convergence_histories: dict = None,
    problem_data: dict = None
):
    """
    Create all types of visualizations for a given problem.
    
    Args:
        results_dict: Performance results for each algorithm
        problem_name: Name of the problem (TSP, Knapsack, etc.)
        output_dir: Directory to save all visualizations
        convergence_histories: Optional convergence histories
        problem_data: Optional problem data for 3D visualization
    """
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating all visualizations for {problem_name}")
    print(f"{'='*60}")
    
    # 1. Performance charts
    print("\n1. Creating performance comparison charts...")
    try:
        create_performance_dashboard(
            results_dict, 
            problem_name,
            output_dir=str(output_path / "performance"),
            convergence_histories=convergence_histories
        )
        print("   ✓ Performance charts created")
    except Exception as e:
        print(f"   ✗ Error creating performance charts: {e}")
    
    # 2. 3D visualization (if problem data provided)
    if problem_data:
        print("\n2. Creating 3D visualizations...")
        try:
            problem_type = problem_name.lower().replace(' ', '_')
            create_3d_visualization_suite(
                problem_type,
                problem_data,
                output_dir=str(output_path / "3d_surface")
            )
            print("   ✓ 3D visualizations created")
        except Exception as e:
            print(f"   ✗ Error creating 3D visualizations: {e}")
    
    print(f"\nAll visualizations saved to: {output_path}")
    return output_path


# Quick demo function
def run_visualization_demo():
    """Run a quick demo of the visualization capabilities."""
    print("\n" + "="*60)
    print("Search_Algorithms Visualization Demo")
    print("="*60)
    print("\nAvailable visualization modules:")
    print("  1. Performance Charts:")
    print("     - plot_performance_bar()")
    print("     - plot_performance_box()")
    print("     - plot_convergence_comparison()")
    print("     - plot_stability_analysis()")
    print("  2. Sensitivity Analysis:")
    print("     - plot_parameter_sensitivity_1d()")
    print("     - plot_parameter_sensitivity_2d()")
    print("     - ParameterSensitivityAnalyzer class")
    print("  3. 3D Visualization:")
    print("     - plot_tsp_objective_surface()")
    print("     - plot_knapsack_objective_surface()")
    print("     - plot_nqueens_objective_surface()")
    print("\nUsage example:")
    print("  from Search_Algorithms.visualization import (")
    print("      create_performance_dashboard,")
    print("      plot_parameter_sensitivity_1d,")
    print("      plot_tsp_objective_surface")
    print("  )")
    print("\n" + "="*60)
