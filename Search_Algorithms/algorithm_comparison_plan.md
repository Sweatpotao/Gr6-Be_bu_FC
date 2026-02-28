# Algorithm Comparison Plan

## Overview
This document outlines meaningful algorithm comparisons for the search and optimization algorithms in the codebase. The comparisons are organized by problem type (continuous vs. discrete) and algorithm category to highlight key trade-offs and performance characteristics.

## Algorithm Categories

The codebase contains algorithms from 5 main categories:
1. **Classical Search Algorithms** - BFS, DFS, UCS, Greedy, A*
2. **Evolutionary Algorithms** - Genetic Algorithm (GA), Differential Evolution (DE)
3. **Human-Based Algorithms** - Teaching-Learning Based Optimization (TLBO)
4. **Local Search Algorithms** - Hill Climbing, Simulated Annealing
5. **Swarm Intelligence Algorithms** - PSO, ABC (Artificial Bee Colony), ACO, Cuckoo Search, Firefly Algorithm

## Comparison Matrix

### 1. Discrete Problems

#### 1.1 Grid Pathfinding
**Best comparisons:**
- **BFS vs. DFS** - Breadth-first vs. Depth-first exploration strategies
- **UCS vs. A*** - Uniform cost search vs. heuristic-based search
- **Greedy vs. A*** - Pure heuristic search vs. heuristic + cost search
- **BFS vs. UCS** - Unweighted vs. weighted shortest path
- **DFS vs. A*** - Brute-force vs. informed search

#### 1.2 N-Queens
**Best comparisons:**
- **BFS vs. DFS** - Complete search vs. depth-first exploration
- **Greedy vs. A*** - Greedy placement vs. heuristic-based search
- **DFS vs. A*** - Uninformed vs. informed search

#### 1.3 Traveling Salesman Problem (TSP)
**Best comparisons:**
- **Greedy vs. A*** - Heuristic construction vs. heuristic-based search
- **GA vs. ACO** - Evolutionary approach vs. swarm-based approach
- **Simulated Annealing vs. Hill Climbing** - Local search with/without random jumps
- **A* vs. Local Search** - Complete search vs. heuristic local optimization

#### 1.4 Knapsack Problem
**Best comparisons:**
- **Greedy vs. Dynamic Programming** - Approximation vs. optimal solution
- **GA vs. DE** - Binary vs. real-valued encoding for discrete optimization
- **Local Search vs. Evolutionary** - Single-solution vs. population-based

### 2. Continuous Optimization Problems

#### 2.1 Sphere Function (Baseline)
**Best comparisons:**
- **All algorithms on simple unimodal function** - Baseline performance comparison
- **Gradient-based vs. Metaheuristics** - When problem is smooth and convex

#### 2.2 Ackley Function
**Best comparisons:**
- **Hill Climbing vs. Simulated Annealing** - Local search with vs. without random jumps
- **GA vs. DE** - Genetic algorithm vs. differential evolution
- **PSO vs. ABC** - Particle swarm vs. artificial bee colony
- **Firefly vs. Cuckoo** - Swarm intelligence variants
- **TLBO vs. GA** - Human-inspired vs. evolutionary approach
- **Local Search vs. Evolutionary** - Single-point vs. population-based

#### 2.3 Rastrigin, Rosenbrock, Griewank Functions
Same comparison pairs as Ackley function, plus:
- **Algorithm performance across different functions** - Test algorithm robustness

## Key Performance Metrics

For each comparison, measure these metrics:
- **Solution Quality (Best Score, Mean Score)** - How good is the solution?
- **Runtime** - How fast does the algorithm find a solution?
- **Effort (Evaluations/Nodes Expanded)** - How many steps does it take?
- **Success Rate** - What percentage of runs find a valid solution?
- **Convergence Rate** - How quickly does it find the optimal solution?

## Spider Web (Radar Chart) Visualization

### Recommended Axes for Spider Web Chart

For a spider/radar chart comparing algorithms, use these 5 axes evenly distributed at 72° intervals:

| Axis | Metric | Calculation | Direction |
|------|--------|-------------|-----------|
| 1 | **Solution Quality** | `1 - (score / worst_score)` | Higher = better |
| 2 | **Speed** | `1 - (time / max_time)` | Higher = faster |
| 3 | **Efficiency** | `1 - (evaluations / max_evaluations)` | Higher = fewer steps |
| 4 | **Reliability** | `success_rate / 100` | Higher = more reliable |
| 5 | **Convergence** | `1 - (iterations_to_converge / max_iterations)` | Higher = faster |

### Spider Web Chart Structure (5 Axes)

```
              Solution Quality
                     |
                     |
        Efficiency   |    Speed
              \      |      /
               \     |     /
                \    |    /
                 \   |   /
                  \  |  /
                   \ | /
        Reliability--+-- Convergence
                     |
```

### How to Create Spider Web Charts

1. **Normalize all metrics** to 0-1 scale for each problem
2. **Each algorithm** is represented as a colored polygon
3. **Larger polygon area** = better overall performance
4. **Compare shapes** to see trade-offs between algorithms

### Example Python Code (Matplotlib)

```python
import numpy as np
import matplotlib.pyplot as plt

def create_spider_chart(algorithms_data, title="Algorithm Comparison"):
    """
    algorithms_data: dict {algorithm_name: [quality, speed, efficiency, reliability, convergence]}
    Each value should be normalized to 0-1 scale
    """
    categories = ['Solution Quality', 'Speed', 'Efficiency', 'Reliability', 'Convergence']
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms_data)))
    
    for idx, (name, values) in enumerate(algorithms_data.items()):
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    return fig
```

### Important Considerations for Spider Charts

**Avoid These Common Mistakes:**
- **Too many algorithms**: Maximum 5-7 algorithms per chart to avoid visual clutter
- **Different scales**: Never mix discrete and continuous algorithms on same chart
- **Unnormalized data**: Always normalize metrics before plotting

**Color Scheme Recommendations:**
| Algorithm Category | Suggested Color |
|-------------------|-----------------|
| Classical (BFS, DFS, A*) | Blues |
| Evolutionary (GA, DE) | Greens |
| Swarm (PSO, ABC, ACO) | Oranges/Reds |
| Local Search (HC, SA) | Purples |
| Human-based (TLBO) | Yellows/Browns |

**When Spider Charts Can Be Misleading:**
- Algorithm A có diện tích lớn hơn B nhưng thua ở metric quan trọng nhất
- Trade-offs không rõ ràng khi các metric có tương quan ngược nhau
- Normalization làm mất đi khoảng cách tuyệt đối giữa các thuật toán

### Chart Types to Generate

| Chart Type | Description | Use Case |
|------------|-------------|----------|
| Per Problem | One chart per problem | Compare all algorithms on Grid Pathfinding, N-Queens, Ackley, etc. |
| Per Category | Group similar algorithms | Classical vs Evolutionary vs Swarm vs Local Search |
| Cross-Problem | Same algorithm across problems | Test algorithm robustness |

### Separate Charts for Discrete vs Continuous Problems

**Discrete Problems (Grid Pathfinding, N-Queens, TSP, Knapsack):**
- Use: BFS, DFS, UCS, Greedy, A*
- Metrics: Path length, nodes expanded, success rate

**Continuous Problems (Ackley, Rastrigin, Rosenbrock, Griewank):**
- Use: Hill Climbing, SA, GA, DE, PSO, ABC, ACO, Cuckoo, Firefly, TLBO
- Metrics: Best score, evaluations, convergence rate

### Metric Mapping for Spider Chart Axes

| Problem Type | Solution Quality | Speed | Efficiency | Reliability | Convergence |
|--------------|-----------------|-------|------------|-------------|-------------|
| **Discrete** | Solution quality score / Path optimality | Runtime (seconds) | Nodes expanded / States visited | Success rate (%) | Steps to first valid solution |
| **Continuous** | `1 / (1 + best_score)` or normalized | Runtime (seconds) | Function evaluations | Success rate within tolerance | Iterations to reach threshold |

**Note**: For continuous problems, lower objective function values are better, so invert for normalization: `quality = 1 / (1 + normalized_score)` or `quality = (max_score - score) / (max_score - min_score)`

### Implementation Notes

- Use matplotlib's `polar()` function or Plotly for interactive charts
- Normalize metrics: `normalized = (value - min) / (max - min)`
- For metrics where lower is better (time, evaluations), invert: `1 - normalized`
- Add algorithm legend with distinct colors
- Use alpha transparency for overlapping polygons

## Recommended Experiment Configurations

### Discrete Problem Experiments
1. **Grid Pathfinding** - Current config has small 4x4 grid; consider larger grids (8x8, 16x16)
2. **N-Queens** - Current config is 8-queens; consider 10-queens, 12-queens
3. **TSP** - Add experiment for Traveling Salesman Problem
4. **Knapsack** - Add experiment for 0-1 Knapsack Problem

### Continuous Problem Experiments
1. **Ackley** - Current: 10 dimensions; consider 20, 30 dimensions
2. **Rastrigin** - Multi-modal test case
3. **Rosenbrock** - Non-convex, narrow valley test case
4. **Griewank** - Multi-modal with wide basins

## Important Considerations

### Comparison Guidelines
- Run each algorithm 10+ times for statistical significance
- Use same parameter settings across comparisons
- Fix random seeds for reproducibility
- Compare algorithms solving identical problem instances

> **Note**: All critical algorithm issues (DFS visited handling, TLBO teacher phase, PSO velocity clamping) have been FIXED in the current codebase, ensuring fair and accurate comparisons.

## Implementation Steps

1. ~~Fix critical algorithm issues (DFS, TLBO, PSO)~~ ✅ (Already Fixed)
2. Expand experiment configurations with more problem sizes
3. Create comparison scripts using existing `experiment_runner.py`
4. Add visualization for convergence curves and performance metrics
5. Run experiments and analyze results

## Expected Insights

From these comparisons, we will gain insights into:
- Which algorithm category performs best for each problem type
- Trade-offs between exploration vs. exploitation
- Effect of problem size on algorithm performance
- Robustness across different problem landscapes
- Convergence characteristics

## Conclusion

These comparison pairs are designed to reveal meaningful differences between algorithms while covering a diverse set of problem types and algorithm approaches. By following this plan, we can systematically evaluate and rank algorithm performance for various search and optimization tasks.