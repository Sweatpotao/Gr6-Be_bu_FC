"""
Visualization module for Search_Algorithms project.

Provides radar/spider chart visualization for:
- Discrete algorithms (BFS, DFS, UCS, Greedy, A*)
- Continuous algorithms (PSO, GA, DE, etc.)
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

from .base import (
    DISCRETE_COLOR_MAP,
    CONTINUOUS_COLORS,
    setup_polar_axes,
    get_algorithm_color,
    normalize_metrics_minmax,
)

__all__ = [
    # Discrete visualization
    'create_spider_chart',
    'create_comparison_grid',
    'create_summary_table',
    # Continuous visualization
    'plot_continuous_radar',
    'plot_radar_chart',
    'normalize_continuous_metrics',
    # Utilities
    'DISCRETE_COLOR_MAP',
    'CONTINUOUS_COLORS',
    'setup_polar_axes',
    'get_algorithm_color',
    'normalize_metrics_minmax',
]

__version__ = '1.0.0'
