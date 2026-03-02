"""
Metaheuristic Discrete Algorithm Comparison Script
Compares metaheuristic optimization algorithms on discrete problems:
- TSP (Traveling Salesman Problem): GA_TSP, HillClimbingTSP, SimulatedAnnealingTSP
- Knapsack: ABC_Knapsack, TLBO_Knapsack

These algorithms inherit from Optimizer base class and use the run() method.
"""

import sys
import time
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import metaheuristic algorithms
from algorithms.evolution.ga_tsp import GeneticAlgorithmTSP
from algorithms.local_search.hill_climbing_tsp import HillClimbingTSP
from algorithms.local_search.simulated_annealing_tsp import SimulatedAnnealingTSP
from algorithms.swarm.abc_knapsack import ABC_Knapsack
from algorithms.human.tlbo_knapsack import TLBO_Knapsack

# Import discrete problems
from problems.discrete.tsp import TSP
from problems.discrete.knapsack import Knapsack


class MetaheuristicDiscreteComparator:
    """Compare metaheuristic optimization algorithms on discrete problems."""
    
    def __init__(self):
        # Algorithm mapping: problem_type -> list of (name, class)
        self.tsp_algorithms = {
            'GA_TSP': GeneticAlgorithmTSP,
            'HillClimbingTSP': HillClimbingTSP,
            'SimulatedAnnealingTSP': SimulatedAnnealingTSP,
        }
        self.knapsack_algorithms = {
            'ABC_Knapsack': ABC_Knapsack,
            'TLBO_Knapsack': TLBO_Knapsack,
        }
        self.results = {}
        
    def run_tsp_comparison(self, n_cities: int = 10, runs: int = 10) -> Dict:
        """Run TSP algorithms comparison."""
        print(f"\n{'='*60}")
        print(f"Problem: TSP (Traveling Salesman Problem)")
        print(f"Cities: {n_cities}, Runs: {runs}")
        print(f"{'='*60}")
        
        problem_results = {}
        
        for algo_name, algo_class in self.tsp_algorithms.items():
            print(f"\n  Running {algo_name}...", end=" ", flush=True)
            
            algo_results = {
                'solutions': [],
                'costs': [],
                'runtimes': [],
                'evaluations': [],
                'histories': [],
                'success_count': 0
            }
            
            for run in range(runs):
                try:
                    # Create fresh TSP instance for each run
                    random.seed(42 + run)
                    tsp_matrix = [[0 if i == j else random.randint(10, 100)
                                   for j in range(n_cities)] for i in range(n_cities)]
                    problem = TSP(distance_matrix=tsp_matrix, start_city=0)
                    
                    # Configure algorithm
                    config = {
                        "timeout": 60,  # 60 seconds timeout
                        "max_iters": 1000 if algo_name == 'GA_TSP' else 5000,
                        "pop_size": 50 if algo_name == 'GA_TSP' else None,
                        "initial_temp": 1000 if algo_name == 'SimulatedAnnealingTSP' else None,
                        "cooling_rate": 0.995 if algo_name == 'SimulatedAnnealingTSP' else None,
                    }
                    # Remove None values
                    config = {k: v for k, v in config.items() if v is not None}
                    
                    # Initialize and run algorithm
                    algo = algo_class(problem, config)
                    start_time = time.time()
                    result = algo.run()
                    runtime = time.time() - start_time
                    
                    # Extract results
                    if result.get('found', False):
                        algo_results['success_count'] += 1
                        algo_results['solutions'].append(result.get('best_solution'))
                        algo_results['costs'].append(result.get('final_score'))
                        algo_results['evaluations'].append(result.get('evaluations', 0))
                        algo_results['histories'].append(result.get('history', []))
                    
                    algo_results['runtimes'].append(runtime)
                    
                except Exception as e:
                    print(f"Error in {algo_name} run {run}: {e}")
                    algo_results['runtimes'].append(float('inf'))
            
            # Calculate summary statistics
            summary = self._calculate_tsp_summary(algo_results, runs)
            problem_results[algo_name] = summary
            
            print(f"[OK] Success: {summary['success_rate']:.1f}%, "
                  f"Avg Cost: {summary['avg_cost']:.2f}, "
                  f"Avg Time: {summary['avg_time']:.4f}s")
        
        self.results['TSP'] = problem_results
        return problem_results
    
    def run_knapsack_comparison(self, n_items: int = 20, runs: int = 10) -> Dict:
        """Run Knapsack algorithms comparison."""
        print(f"\n{'='*60}")
        print(f"Problem: Knapsack")
        print(f"Items: {n_items}, Runs: {runs}")
        print(f"{'='*60}")
        
        problem_results = {}
        
        for algo_name, algo_class in self.knapsack_algorithms.items():
            print(f"\n  Running {algo_name}...", end=" ", flush=True)
            
            algo_results = {
                'solutions': [],
                'values': [],
                'runtimes': [],
                'evaluations': [],
                'histories': [],
                'success_count': 0
            }
            
            for run in range(runs):
                try:
                    # Create fresh Knapsack instance for each run
                    random.seed(42 + run)
                    values = [random.randint(10, 100) for _ in range(n_items)]
                    weights = [random.randint(5, 50) for _ in range(n_items)]
                    capacity = sum(weights) // 2
                    problem = Knapsack(values=values, weights=weights, capacity=capacity)
                    
                    # Configure algorithm
                    config = {
                        "timeout": 60,
                        "max_iters": 500,
                        "pop_size": 50 if algo_name == 'ABC_Knapsack' else 40,
                        "penalty_coef": 1000,
                    }
                    
                    # Initialize and run algorithm
                    algo = algo_class(problem, config)
                    start_time = time.time()
                    result = algo.run()
                    runtime = time.time() - start_time
                    
                    # Extract results
                    if result.get('found', False):
                        algo_results['success_count'] += 1
                        algo_results['solutions'].append(result.get('best_solution'))
                        # For knapsack, cost is negative value (minimization)
                        # We want to report the actual value
                        cost = result.get('final_score')
                        value = -cost if cost is not None else 0
                        algo_results['values'].append(value)
                        algo_results['evaluations'].append(result.get('evaluations', 0))
                        algo_results['histories'].append(result.get('history', []))
                    
                    algo_results['runtimes'].append(runtime)
                    
                except Exception as e:
                    print(f"Error in {algo_name} run {run}: {e}")
                    algo_results['runtimes'].append(float('inf'))
            
            # Calculate summary statistics
            summary = self._calculate_knapsack_summary(algo_results, runs)
            problem_results[algo_name] = summary
            
            print(f"[OK] Success: {summary['success_rate']:.1f}%, "
                  f"Avg Value: {summary['avg_value']:.2f}, "
                  f"Avg Time: {summary['avg_time']:.4f}s")
        
        self.results['Knapsack'] = problem_results
        return problem_results
    
    def _calculate_tsp_summary(self, results: Dict, total_runs: int) -> Dict:
        """Calculate summary statistics for TSP algorithm."""
        valid_times = [t for t in results['runtimes'] if t != float('inf')]
        valid_costs = [c for c in results['costs'] if c is not None]
        
        summary = {
            'success_rate': (results['success_count'] / total_runs) * 100,
            'avg_time': sum(valid_times) / len(valid_times) if valid_times else float('inf'),
            'min_time': min(valid_times) if valid_times else float('inf'),
            'max_time': max(valid_times) if valid_times else float('inf'),
            'avg_cost': sum(valid_costs) / len(valid_costs) if valid_costs else float('inf'),
            'best_cost': min(valid_costs) if valid_costs else float('inf'),
            'worst_cost': max(valid_costs) if valid_costs else float('inf'),
            'avg_evaluations': sum(results['evaluations']) / total_runs if results['evaluations'] else 0,
        }
        
        return summary
    
    def _calculate_knapsack_summary(self, results: Dict, total_runs: int) -> Dict:
        """Calculate summary statistics for Knapsack algorithm."""
        valid_times = [t for t in results['runtimes'] if t != float('inf')]
        valid_values = [v for v in results['values'] if v is not None]
        
        summary = {
            'success_rate': (results['success_count'] / total_runs) * 100,
            'avg_time': sum(valid_times) / len(valid_times) if valid_times else float('inf'),
            'min_time': min(valid_times) if valid_times else float('inf'),
            'max_time': max(valid_times) if valid_times else float('inf'),
            'avg_value': sum(valid_values) / len(valid_values) if valid_values else 0,
            'best_value': max(valid_values) if valid_values else 0,
            'worst_value': min(valid_values) if valid_values else 0,
            'avg_evaluations': sum(results['evaluations']) / total_runs if results['evaluations'] else 0,
        }
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a formatted comparison report."""
        report = []
        report.append("# Metaheuristic Discrete Algorithm Comparison Report\n")
        report.append("=" * 80)
        
        for problem_name, problem_results in self.results.items():
            report.append(f"\n## {problem_name}\n")
            report.append("-" * 80)
            
            if problem_name == 'TSP':
                report.append(f"{'Algorithm':<20} {'Success %':<10} {'Avg Cost':<12} {'Avg Time':<12} {'Evals':<10}")
            else:  # Knapsack
                report.append(f"{'Algorithm':<20} {'Success %':<10} {'Avg Value':<12} {'Avg Time':<12} {'Evals':<10}")
            
            report.append("-" * 80)
            
            # Sort by success rate, then by metric
            if problem_name == 'TSP':
                sorted_results = sorted(
                    problem_results.items(),
                    key=lambda x: (-x[1]['success_rate'], x[1]['avg_cost'])
                )
            else:
                sorted_results = sorted(
                    problem_results.items(),
                    key=lambda x: (-x[1]['success_rate'], -x[1]['avg_value'])
                )
            
            for algo_name, stats in sorted_results:
                if problem_name == 'TSP':
                    metric_val = stats['avg_cost']
                    metric_str = f"{metric_val:>10.2f}"
                else:
                    metric_val = stats['avg_value']
                    metric_str = f"{metric_val:>10.2f}"
                
                report.append(
                    f"{algo_name:<20} "
                    f"{stats['success_rate']:>8.1f}%  "
                    f"{metric_str}  "
                    f"{stats['avg_time']:>10.4f}s  "
                    f"{stats['avg_evaluations']:>8.0f}"
                )
            
            report.append("")
        
        return "\n".join(report)
    
    def generate_spider_data(self) -> Dict:
        """Generate normalized data for spider/radar charts."""
        spider_data = {}
        
        for problem_name, problem_results in self.results.items():
            spider_data[problem_name] = {}
            
            # Find max values for normalization
            max_time = max(r['avg_time'] for r in problem_results.values() 
                          if r['avg_time'] != float('inf'))
            max_evals = max(r['avg_evaluations'] for r in problem_results.values())
            
            for algo_name, stats in problem_results.items():
                # Normalize to 0-1 scale (higher is better)
                if problem_name == 'TSP':
                    # For TSP: lower cost is better
                    costs = [r['avg_cost'] for r in problem_results.values() 
                            if r['avg_cost'] != float('inf')]
                    min_cost, max_cost = min(costs), max(costs)
                    cost_range = max_cost - min_cost if max_cost > min_cost else 1
                    quality = 1 - ((stats['avg_cost'] - min_cost) / cost_range) if stats['avg_cost'] != float('inf') else 0
                else:
                    # For Knapsack: higher value is better
                    values = [r['avg_value'] for r in problem_results.values()]
                    min_val, max_val = min(values), max(values)
                    val_range = max_val - min_val if max_val > min_val else 1
                    quality = (stats['avg_value'] - min_val) / val_range if val_range > 0 else 1
                
                normalized = {
                    'solution_quality': max(0, min(1, quality)),
                    'speed': 1 - (stats['avg_time'] / max_time) if max_time > 0 and stats['avg_time'] != float('inf') else 0,
                    'efficiency': 1 - (stats['avg_evaluations'] / max_evals) if max_evals > 0 else 0,
                    'reliability': stats['success_rate'] / 100,
                    'convergence': stats['success_rate'] / 100
                }
                
                spider_data[problem_name][algo_name] = [
                    normalized['solution_quality'],
                    normalized['speed'],
                    normalized['efficiency'],
                    normalized['reliability'],
                    normalized['convergence']
                ]
        
        return spider_data
    
    def save_spider_chart_data(self, output_file: str = "metaheuristic_spider_data.json"):
        """Save spider chart data to JSON file."""
        spider_data = self.generate_spider_data()
        
        with open(output_file, 'w') as f:
            json.dump(spider_data, f, indent=2)
        
        print(f"\nSpider chart data saved to: {output_file}")


def main():
    """Main entry point for metaheuristic discrete algorithm comparison."""
    print("Metaheuristic Discrete Algorithm Comparison")
    print("TSP Algorithms: GA_TSP, HillClimbingTSP, SimulatedAnnealingTSP")
    print("Knapsack Algorithms: ABC_Knapsack, TLBO_Knapsack")
    
    comparator = MetaheuristicDiscreteComparator()
    
    # Run TSP comparison
    comparator.run_tsp_comparison(n_cities=10, runs=10)
    
    # Run Knapsack comparison
    comparator.run_knapsack_comparison(n_items=20, runs=10)
    
    # Generate and print report
    report = comparator.generate_report()
    print("\n" + report)
    
    # Save report to file
    report_file = "metaheuristic_discrete_comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    
    # Save spider chart data
    comparator.save_spider_chart_data("metaheuristic_spider_data.json")
    
    # Generate visualization
    try:
        from visualization.discrete_comparison import create_comparison_grid
        spider_data = comparator.generate_spider_data()
        create_comparison_grid(spider_data, output_dir="comparison_charts")
        print("\nComparison charts saved to: comparison_charts/")
    except Exception as e:
        print(f"\nNote: Could not generate charts: {e}")
    
    print("\n" + "="*60)
    print("Metaheuristic Comparison Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
