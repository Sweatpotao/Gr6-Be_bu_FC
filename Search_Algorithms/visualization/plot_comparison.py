#!/usr/bin/env python3
"""
Backward compatibility wrapper for continuous comparison visualization.

NOTE: This module is deprecated. Please import from the new location:
    from visualization.continuous_comparison import plot_continuous_radar
"""

import warnings

warnings.warn(
    "plot_comparison is deprecated. "
    "Use 'visualization.continuous_comparison' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from the new module
from .continuous_comparison import (
    plot_continuous_radar,
    plot_continuous_comparison_legacy,
    normalize_continuous_metrics,
    normalize_metrics_legacy,
    plot_radar_chart,
)

__all__ = [
    'plot_continuous_radar',
    'plot_continuous_comparison_legacy',
    'normalize_continuous_metrics',
    'normalize_metrics_legacy',
    'plot_radar_chart',
]
