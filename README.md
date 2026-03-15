# Search & Optimization Algorithms Comparison Framework

A comprehensive Python framework for comparing classical search algorithms and metaheuristic optimization algorithms on various benchmark problems. This framework supports 5 algorithm categories with 20+ algorithms across 10 different problems (5 continuous + 5 discrete).

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Supported Algorithms](#-supported-algorithms)
- [Supported Problems](#-supported-problems)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Visualization](#-visualization)
- [Performance Metrics](#-performance-metrics)
- [Output Structure](#-output-structure)
- [References](#-references)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏗️ **Modular Architecture** | Registry-based system for easy algorithm and problem registration |
| 📄 **YAML Configuration** | Declarative experiment setup with flexible configuration |
| 🔄 **Multiple Runs** | Statistical significance with configurable number of runs |
| ⏱️ **Timeout Handling** | Automatic termination after configurable timeout |
| 📊 **Rich Visualization** | Spider/radar charts, comparison grids, 3D surfaces |
| 📈 **Comprehensive Metrics** | Best score, mean, std, time, evaluations, success rate |
| 🧪 **Problem Cloning** | Fresh problem instances for each independent run |
| 🔍 **Sensitivity Analysis** | Parameter sensitivity analysis for metaheuristics |

---

## 📁 Project Structure

```
Search_Algorithms/
├── algorithms/              # Algorithm implementations
│   ├── base/               # Base classes (Optimizer, SearchAlgorithm)
│   ├── classical/          # BFS, DFS, UCS, Greedy, A*
│   ├── evolution/          # GA, DE, GA_TSP
│   ├── human/              # TLBO, TLBO_Knapsack
│   ├── local_search/       # Hill Climbing, Simulated Annealing
│   └── swarm/              # PSO, ABC, ACO, Cuckoo, Firefly
├── problems/               # Problem definitions
│   ├── continuous/         # Sphere, Ackley, Rastrigin, Rosenbrock, Griewank
│   └── discrete/           # TSP, Knapsack, N-Queens, Grid Pathfinding, Graph Coloring
├── visualization/          # Visualization modules
│   ├── discrete_comparison.py    # Spider charts for discrete
│   ├── continuous_comparison.py  # Radar charts for continuous
│   └── plot_comparison.py        # General plotting utilities
├── experiment/             # Experiment framework
│   ├── experiment_runner.py      # Core runner
│   ├── registry.py               # Algorithm/Problem registry
│   ├── logger.py                 # Results logging
│   └── run_experiment.py         # Entry point
├── config/                 # YAML configuration files
├── data/                   # Output results
│   ├── charts_from_results/     # Generated charts
│   ├── plots/                    # Visualization plots
│   ├── summary_results/          # Text summaries
│   └── sensitivity_analysis/     # Parameter sensitivity
└── comparison_charts/      # Final comparison visualizations
```

---

## 🧮 Supported Algorithms

### 1. Classical Search (Discrete Only)

| Algorithm | File | Description |
|-----------|------|-------------|
| **BFS** | `bfs.py` | Breadth-First Search - Complete, uninformed search |
| **DFS** | `dfs.py` | Depth-First Search with depth limit |
| **UCS** | `ucs.py` | Uniform Cost Search - Optimal for weighted graphs |
| **Greedy** | `greedy.py` | Greedy Best-First Search |
| **A*** | `astar.py` | A-Star with admissible heuristics |

### 2. Local Search

| Algorithm | File | Description |
|-----------|------|-------------|
| **Hill Climbing** | `hill_climbing.py` | Local optimization with neighbor selection |
| **HillClimbingTSP** | `hill_climbing_tsp.py` | TSP-specific with 2-opt neighborhood |
| **Simulated Annealing** | `simulated_annealing.py` | Local search with temperature-based acceptance |
| **SimulatedAnnealingTSP** | `simulated_annealing_tsp.py` | TSP-specific with 2-opt |

### 3. Evolutionary Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **Genetic Algorithm (GA)** | `genetic_algorithm.py` | Selection, crossover, mutation |
| **Differential Evolution (DE)** | `differential_evolution.py` | Vector-based mutation strategy |
| **GA_TSP** | `ga_tsp.py` | GA specialized for TSP with permutation encoding |

### 4. Swarm Intelligence

| Algorithm | File | Description |
|-----------|------|-------------|
| **PSO** | `pso.py` | Particle Swarm Optimization |
| **ABC** | `abc.py` | Artificial Bee Colony |
| **ABC_Knapsack** | `abc_knapsack.py` | ABC for Knapsack problem |
| **ACO** | `aco.py` | Ant Colony Optimization (ACOR for continuous) |
| **ACO_Discrete** | `aco_discrete.py` | ACO for discrete problems |
| **Cuckoo Search** | `cuckoo.py` | Cuckoo search via Lévy flights |
| **Firefly Algorithm** | `firefly.py` | Firefly-inspired optimization |

### 5. Human-Based Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **TLBO** | `tlbo.py` | Teaching-Learning Based Optimization |
| **TLBO_Knapsack** | `tlbo_knapsack.py` | TLBO for Knapsack problem |

---

## 🎯 Supported Problems

### Continuous Problems (Benchmark Functions)

| Problem | Global Minimum | Domain | Characteristics |
|---------|----------------|--------|-----------------|
| **Sphere** | f(0,...,0) = 0 | [-5.12, 5.12]ᵈ | Unimodal, convex, baseline function |
| **Ackley** | f(0,...,0) = 0 | [-32.768, 32.768]ᵈ | Many local minima, "volcano" shape |
| **Rastrigin** | f(0,...,0) = 0 | [-5.12, 5.12]ᵈ | Highly multi-modal |
| **Rosenbrock** | f(1,...,1) = 0 | [-2.048, 2.048]ᵈ | Narrow valley, non-convex |
| **Griewank** | f(0,...,0) = 0 | [-600, 600]ᵈ | Multi-modal with wide basins |

### Discrete Problems

| Problem | Description | State Representation |
|---------|-------------|---------------------|
| **TSP** | Traveling Salesman Problem | (current_city, visited_bitmask) |
| **Knapsack** | 0/1 Knapsack Problem | (index, current_weight) |
| **N-Queens** | Place N queens without conflicts | Tuple of queen positions |
| **Grid Pathfinding** | Find path on grid with obstacles | (x, y) coordinates |
| **Graph Coloring** | Color graph with minimum colors | Vertex color assignments |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install numpy matplotlib pyyaml
```

### Clone Repository

```bash
git clone <repository-url>
cd Gr6-Be_bu_FC
```

---

## 🚀 Quick Start

### 1. Run Metaheuristic Comparison (Discrete)

```bash
cd Search_Algorithms
python run_metaheuristic_discrete_comparison.py
```

This runs comparison for:
- **TSP**: GA_TSP, HillClimbingTSP, SimulatedAnnealingTSP
- **Knapsack**: ABC_Knapsack, TLBO_Knapsack

### 2. Run Classical Search Comparison

```bash
cd Search_Algorithms
python run_discrete_comparison.py
```

This runs comparison for:
- **Grid Pathfinding**, **N-Queens**, **TSP**, **Knapsack**
- Algorithms: BFS, DFS, UCS, Greedy, A*

### 3. Run All Experiments

```bash
cd Search_Algorithms
python run_all_experiments.py
```

This runs all configured experiments from YAML files.

---

## 📖 Usage

### Using Configuration Files

Create a YAML configuration file:

```yaml
# config/my_experiment.yaml
experiment:
  runs: 30

problem:
  name: tsp
  params:
    distance_matrix:
      - [0, 29, 20, 21]
      - [29, 0, 15, 29]
      - [20, 15, 0, 15]
      - [21, 29, 15, 0]
    start_city: 0

algorithms:
  GA_TSP:
    timeout: 60
    max_iters: 500
    pop_size: 50
    mutation_rate: 0.2
    crossover_rate: 0.9
  
  HillClimbingTSP:
    timeout: 60
    max_iters: 1000
    n_neighbors: 50
    two_opt: true
```

Run the experiment:

```python
from experiment.run_experiment import run_experiment

results = run_experiment("config/my_experiment.yaml")

for algo_name, summary in results.items():
    print(f"{algo_name}:")
    print(f"  Best Score: {summary['best_score']}")
    print(f"  Mean Time: {summary['mean_time']:.4f}s")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
```

### Direct Algorithm Usage

```python
from algorithms.swarm.pso import PSO
from problems.continuous.sphere import SphereFunction

# Create problem
problem = SphereFunction(dim=10)

# Configure algorithm
config = {
    "n_particles": 30,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5,
    "max_iters": 1000,
    "timeout": 60
}

# Run algorithm
algo = PSO(problem, config)
result = algo.run()

print(f"Best solution: {result['best_solution']}")
print(f"Best fitness: {result['final_score']}")
print(f"Evaluations: {result['evaluations']}")
```

---

## ⚙️ Configuration

### Available Configuration Options

#### Common Parameters (All Algorithms)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | int | 300 | Maximum runtime in seconds |
| `max_evaluations` | int | inf | Maximum function evaluations |

#### Algorithm-Specific Parameters

**PSO:**
```yaml
PSO:
  n_particles: 30      # Swarm size
  w: 0.7               # Inertia weight
  c1: 1.5              # Cognitive coefficient
  c2: 1.5              # Social coefficient
  max_iters: 1000      # Maximum iterations
```

**Genetic Algorithm:**
```yaml
GA:
  pop_size: 50         # Population size
  mutation_rate: 0.1   # Mutation probability
  crossover_rate: 0.9  # Crossover probability
  elite_size: 2        # Number of elite individuals
  max_iters: 1000      # Maximum iterations
```

**Simulated Annealing:**
```yaml
SimulatedAnnealing:
  initial_temp: 100    # Initial temperature
  cooling_rate: 0.95   # Cooling rate
  max_iters: 5000      # Maximum iterations
```

**ABC (Artificial Bee Colony):**
```yaml
ABC:
  pop_size: 50         # Number of food sources
  limit: 100           # Limit for scout bees
  max_iters: 1000      # Maximum iterations
```

---

## 📊 Visualization

### Spider/Radar Charts

Spider charts compare algorithms across 5 metrics:
- **Solution Quality**: How good is the solution?
- **Speed**: How fast is the algorithm?
- **Efficiency**: How many evaluations/steps?
- **Reliability**: Success rate percentage
- **Convergence**: How quickly does it converge?

### Generate Visualizations

```python
from visualization.discrete_comparison import create_comparison_grid

# After running experiments
spider_data = comparator.generate_spider_data()
create_comparison_grid(spider_data, output_dir="comparison_charts")
```

### 3D Surface Plots (Continuous Problems)

```python
from visualize_3d_model import generate_3d_surface_from_problem

# Generate 3D surface for Ackley function
generate_3d_surface_from_problem("ackley", dim=2, resolution=100)
```

---

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Best Score** | Best solution found across all runs |
| **Mean Score** | Average solution quality |
| **Std Score** | Standard deviation of scores |
| **Mean Time** | Average runtime in seconds |
| **Mean Effort** | Average nodes expanded (discrete) or evaluations (continuous) |
| **Success Rate** | Percentage of runs finding valid solution |
| **Timeout Count** | Number of runs that timed out |

---

## 📂 Output Structure

After running experiments:

```
Search_Algorithms/
├── comparison_charts/
│   ├── discrete_comparison_grid.png     # All discrete problems
│   ├── tsp_spider.png                   # TSP-specific chart
│   ├── knapsack_spider.png              # Knapsack chart
│   └── discrete_summary.txt             # Text summary
├── data/
│   ├── charts_from_results/
│   │   ├── ACKLEY/
│   │   │   ├── 3d_surface/
│   │   │   └── performance/
│   │   └── ... (other problems)
│   ├── plots/
│   │   ├── ackley_radar.png
│   │   └── ... (other plots)
│   ├── summary_results/
│   │   ├── ackley_results.txt
│   │   └── ... (other results)
│   └── sensitivity_analysis/
│       ├── ackley/
│       └── ... (other problems)
└── [experiment]_results.txt              # Experiment results
```

---

## 🔧 Algorithm Registry

Algorithms and problems are registered in `experiment/registry.py`:

```python
from experiment.registry import PROBLEM_REGISTRY, ALGORITHM_REGISTRY

# List available problems
print(PROBLEM_REGISTRY.keys())
# ['sphere', 'ackley', 'rastrigin', 'rosenbrock', 'griewank',
#  'tsp', 'knapsack', 'n_queens', 'grid_pathfinding', 'graph_coloring']

# List available algorithms
print(ALGORITHM_REGISTRY.keys())
# ['BFS', 'DFS', 'UCS', 'Greedy', 'AStar',
#  'HillClimbing', 'SimulatedAnnealing', 'GA', 'DE',
#  'PSO', 'ABC', 'ACO', 'CuckooSearch', 'FireflyAlgorithm', 'TLBO', ...]
```

---

## 📝 Base Classes

### For Metaheuristic Algorithms

```python
from algorithms.base.optimizer_base import Optimizer

class MyAlgorithm(Optimizer):
    def run(self):
        # Initialize population/solution
        # Main optimization loop
        # Update best_solution and best_fitness
        return self._build_result()
```

### For Classical Search

```python
from algorithms.base.search_base import SearchAlgorithm

class MySearch(SearchAlgorithm):
    def search(self):
        # Initialize
        # Search loop with get_neighbors(), is_goal()
        # Track parent_map for path reconstruction
        return self._build_result()
```

---

## 💡 Tips & Best Practices

1. **Fix Random Seeds**: For reproducible results, set random seeds before running
2. **Multiple Runs**: Use at least 10-30 runs for statistical significance
3. **Timeout**: Set appropriate timeout based on problem complexity
4. **Parameter Tuning**: Use sensitivity analysis to find optimal parameters
5. **Problem Selection**: 
   - Classical search for small discrete problems
   - Metaheuristics for large-scale and continuous problems

---

## 📚 References

This framework implements algorithms from the following domains:

### Classical Search
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)

### Evolutionary Computation
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*
- Storn, R., & Price, K. (1997). Differential Evolution

### Swarm Intelligence
- Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization
- Karaboga, D. (2005). Artificial Bee Colony Algorithm
- Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*
- Yang, X. S. (2009). Firefly Algorithm
- Yang, X. S., & Deb, S. (2009). Cuckoo Search

### Human-Based Optimization
- Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011). Teaching-Learning-Based Optimization

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or suggestions, please open an issue on the repository.

---

**Note**: This framework was developed for educational and research purposes in algorithm comparison and optimization studies.