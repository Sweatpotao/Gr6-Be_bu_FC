"""
Discrete Algorithm Comparison Script
Compares classical search algorithms on discrete problems:
- Grid Pathfinding
- N-Queens
- TSP (Traveling Salesman Problem)
- Knapsack

Algorithms: BFS, DFS, UCS, Greedy, A*
"""

import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.classical.bfs import BFS
from algorithms.classical.dfs import DFS
from algorithms.classical.ucs import UCS
from algorithms.classical.greedy import Greedy
from algorithms.classical.astar import AStar

from problems.discrete.grid_pathfinding import GridPathfinding
from problems.discrete.n_queens import NQueens
from problems.discrete.tsp import TSP
from problems.discrete.knapsack import Knapsack


class DiscreteComparator:
    """Compare discrete search algorithms across multiple problems."""
    
    def __init__(self):
        self.algorithms = {
            'BFS': BFS,
            'DFS': DFS,
            'UCS': UCS,
            'Greedy': Greedy,
            'A*': AStar
        }
        self.results = {}
        
    def run_comparison(self, problem_name: str, problem_instance, runs: int = 10) -> Dict:
        """Run all algorithms on a single problem."""
        print(f"\n{'='*60}")
        print(f"Problem: {problem_name}")
        print(f"{'='*60}")
        
        problem_results = {}
        
        for algo_name, algo_class in self.algorithms.items():
            print(f"\n  Running {algo_name}...", end=" ", flush=True)
            
            algo_results = {
                'solutions': [],
                'runtimes': [],
                'nodes_expanded': [],
                'success_count': 0
            }
            
            for run in range(runs):
                problem = None
                try:
                    # Create fresh problem instance for each run
                    if problem_name == "Grid Pathfinding":
                        grid = [[0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0],
                                [0, 1, 0, 0]]
                        problem = GridPathfinding(grid, start=(0, 0), goal=(3, 3))
                    elif problem_name == "N-Queens":
                        problem = NQueens(n=8)
                    elif problem_name == "TSP":
                        n_cities = 5
                        tsp_matrix = [[0 if i == j else random.randint(10, 100) 
                                       for j in range(n_cities)] for i in range(n_cities)]
                        problem = TSP(distance_matrix=tsp_matrix, start_city=0)
                    elif problem_name == "Knapsack":
                        n_items = 10
                        values = [random.randint(10, 100) for _ in range(n_items)]
                        weights = [random.randint(5, 50) for _ in range(n_items)]
                        capacity = sum(weights) // 2
                        problem = Knapsack(values=values, weights=weights, capacity=capacity)
                    else:
                        problem = problem_instance
                    
                    # Initialize algorithm with timeout config
                    algo_config = {"timeout": 60}  # 60 seconds timeout per run
                    algo = algo_class(problem, algo_config)
                    
                    # Run algorithm
                    start_time = time.time()
                    result = algo.search()
                    runtime = time.time() - start_time
                    
                    # Extract solution from result dict
                    solution = None
                    if isinstance(result, dict):
                        if result.get('found'):
                            solution = result.get('solution')
                    else:
                        solution = result
                    
                    # Collect metrics
                    if solution is not None:
                        algo_results['success_count'] += 1
                        algo_results['solutions'].append(solution)
                    
                    algo_results['runtimes'].append(runtime)
                    
                    # Get nodes expanded if available
                    nodes = getattr(algo, 'nodes_expanded', 0)
                    algo_results['nodes_expanded'].append(nodes)
                    
                except Exception as e:
                    print(f"Error in {algo_name}: {e}")
                    solution = None  # Fix UnboundLocalError
                    algo_results['runtimes'].append(float('inf'))
                    algo_results['nodes_expanded'].append(0)
            
            # Calculate summary statistics
            summary = self._calculate_summary(algo_results, problem)
            problem_results[algo_name] = summary
            
            print(f"[OK] Success: {summary['success_rate']:.1f}%, "
                  f"Avg Time: {summary['avg_time']:.4f}s")
        
        self.results[problem_name] = problem_results
        return problem_results
    
    def _calculate_summary(self, results: Dict, problem) -> Dict:
        """Calculate summary statistics for an algorithm."""
        runs = len(results['runtimes'])
        
        # Filter out failed runs for time calculation
        valid_times = [t for t in results['runtimes'] if t != float('inf')]
        
        summary = {
            'success_rate': (results['success_count'] / runs) * 100,
            'avg_time': sum(valid_times) / len(valid_times) if valid_times else float('inf'),
            'min_time': min(valid_times) if valid_times else float('inf'),
            'max_time': max(valid_times) if valid_times else float('inf'),
            'avg_nodes': sum(results['nodes_expanded']) / runs,
            'solution_quality': self._evaluate_solution_quality(results['solutions'], problem)
        }
        
        return summary
    
    def _evaluate_solution_quality(self, solutions: List, problem) -> float:
        """Evaluate solution quality based on problem type."""
        if not solutions:
            return 0.0
        
        # For pathfinding: shorter path is better
        # For N-Queens: any valid solution is equally good
        # For TSP: shorter tour is better
        # For Knapsack: higher value is better
        
        if hasattr(problem, 'evaluate_solution'):
            scores = [problem.evaluate_solution(sol) for sol in solutions]
            return sum(scores) / len(scores)
        
        return float(len(solutions))  # Default: count successful solutions
    
    def generate_report(self) -> str:
        """Generate a formatted comparison report."""
        report = []
        report.append("# Discrete Algorithm Comparison Report\n")
        report.append("=" * 80)
        
        for problem_name, problem_results in self.results.items():
            report.append(f"\n## {problem_name}\n")
            report.append("-" * 80)
            report.append(f"{'Algorithm':<12} {'Success %':<10} {'Avg Time':<12} {'Avg Nodes':<12} {'Quality':<10}")
            report.append("-" * 80)
            
            # Sort by success rate, then by avg time
            sorted_results = sorted(
                problem_results.items(),
                key=lambda x: (-x[1]['success_rate'], x[1]['avg_time'])
            )
            
            for algo_name, stats in sorted_results:
                report.append(
                    f"{algo_name:<12} "
                    f"{stats['success_rate']:>8.1f}%  "
                    f"{stats['avg_time']:>10.4f}s  "
                    f"{stats['avg_nodes']:>10.1f}   "
                    f"{stats['solution_quality']:>8.2f}"
                )
            
            report.append("")
        
        return "\n".join(report)
    
    def generate_spider_data(self) -> Dict:
        """Generate normalized data for spider/radar charts."""
        spider_data = {}
        
        for problem_name, problem_results in self.results.items():
            spider_data[problem_name] = {}
            
            # Find max values for normalization
            max_time = max(r['avg_time'] for r in problem_results.values() if r['avg_time'] != float('inf'))
            max_nodes = max(r['avg_nodes'] for r in problem_results.values())
            
            for algo_name, stats in problem_results.items():
                # Normalize to 0-1 scale (higher is better)
                normalized = {
                    'solution_quality': min(stats['solution_quality'] / 100, 1.0) if stats['solution_quality'] > 1 else stats['solution_quality'],
                    'speed': 1 - (stats['avg_time'] / max_time) if max_time > 0 and stats['avg_time'] != float('inf') else 0,
                    'efficiency': 1 - (stats['avg_nodes'] / max_nodes) if max_nodes > 0 else 0,
                    'reliability': stats['success_rate'] / 100,
                    'convergence': stats['success_rate'] / 100  # Using success rate as proxy
                }
                spider_data[problem_name][algo_name] = [
                    normalized['solution_quality'],
                    normalized['speed'],
                    normalized['efficiency'],
                    normalized['reliability'],
                    normalized['convergence']
                ]
        
        return spider_data
    
    def save_spider_chart_data(self, output_file: str = "discrete_spider_data.json"):
        """Save spider chart data to JSON file."""
        spider_data = self.generate_spider_data()
        
        with open(output_file, 'w') as f:
            json.dump(spider_data, f, indent=2)
        
        print(f"\nSpider chart data saved to: {output_file}")


def main():
    """Main entry point for discrete algorithm comparison."""
    print("Discrete Algorithm Comparison")
    print("Algorithms: BFS, DFS, UCS, Greedy, A*")
    print("Problems: Grid Pathfinding, N-Queens, TSP, Knapsack")
    
    comparator = DiscreteComparator()
    
    # Run comparisons on each problem
    # Grid Pathfinding (4x4 grid)
    grid = [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0]]
    grid_problem = GridPathfinding(grid, start=(0, 0), goal=(3, 3))
    comparator.run_comparison("Grid Pathfinding", grid_problem, runs=5)
    
    # N-Queens (5 queens - reduced from 8 for faster execution)
    nqueens_problem = NQueens(n=5)
    comparator.run_comparison("N-Queens", nqueens_problem, runs=5)
    
    # TSP (5 cities - smaller for faster execution)
    import random
    random.seed(42)
    n_cities = 5
    tsp_matrix = [[0 if i == j else random.randint(10, 100)
                   for j in range(n_cities)] for i in range(n_cities)]
    tsp_problem = TSP(distance_matrix=tsp_matrix, start_city=0)
    comparator.run_comparison("TSP", tsp_problem, runs=5)
    
    # Knapsack (10 items)
    n_items = 10
    values = [random.randint(10, 100) for _ in range(n_items)]
    weights = [random.randint(5, 50) for _ in range(n_items)]
    capacity = sum(weights) // 2
    knapsack_problem = Knapsack(values=values, weights=weights, capacity=capacity)
    comparator.run_comparison("Knapsack", knapsack_problem, runs=5)
    
    # Generate and print report
    report = comparator.generate_report()
    print("\n" + report)
    
    # Save report to file
    report_file = "discrete_comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    
    # Save spider chart data
    comparator.save_spider_chart_data("discrete_spider_data.json")
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
