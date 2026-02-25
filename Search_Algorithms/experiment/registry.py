# continuous
from problems.continuous.sphere import SphereFunction
from problems.continuous.ackley import Ackley
from problems.continuous.rastrigin import Rastrigin
from problems.continuous.rosenbrock import Rosenbrock
from problems.continuous.griewank import Griewank
# discrete
from problems.discrete.graph_coloring import GraphColoring
from problems.discrete.grid_pathfinding import GridPathfinding
from problems.discrete.knapsack import Knapsack
from problems.discrete.n_queens import NQueens
from problems.discrete.tsp import TSP

#  classical
from algorithms.classical.bfs import BFS
from algorithms.classical.astar import AStar
from algorithms.classical.dfs import DFS
from algorithms.classical.ucs import UCS
from algorithms.classical.greedy import Greedy
# local_search
from algorithms.local_search.hill_climbing import HillClimbing
from algorithms.local_search.simulated_annealing import SimulatedAnnealing
# evolution
from algorithms.evolution.differential_evolution import DifferentialEvolution
from algorithms.evolution.genetic_algorithm import GeneticAlgorithm
# swarm
from algorithms.swarm.abc import ABC
from algorithms.swarm.aco import ACO
from algorithms.swarm.cuckoo import CuckooSearch
from algorithms.swarm.firefly import FireflyAlgorithm
from algorithms.swarm.pso import PSO
# human
from algorithms.human.tlbo import TLBO

PROBLEM_REGISTRY = {
    # Continuous
    "sphere": SphereFunction,
    "ackley": Ackley,
    "rastrigin": Rastrigin,
    "rosenbrock": Rosenbrock,
    "griewank": Griewank,
    # Discrete
    "graph_coloring": GraphColoring,
    "grid_pathfinding": GridPathfinding,
    "knapsack": Knapsack,
    "n_queens": NQueens,
    "tsp": TSP
}

ALGORITHM_REGISTRY = {
    # Classical
    "BFS": BFS,                                     # Discrete
    "DFS": DFS,                                     # Discrete
    "UCS": UCS,                                     # Discrete
    "Greedy": Greedy,                               # Discrete
    "AStar": AStar,                                 # Discrete
    # Local search
    "HillClimbing": HillClimbing,                   # Discrete & Continuous
    "SimulatedAnnealing": SimulatedAnnealing,       # Discrete & Continuous
    # Evolution
    "GA": GeneticAlgorithm,                         # Discrete & Continuous
    "DE": DifferentialEvolution,                    # Continuous
    # Swarm
    "ACO": ACO,                                     # Discrete
    "ABC": ABC,                                     # Continuous
    "CuckooSearch": CuckooSearch,                   # Continuous
    "FireflyAlgorithm": FireflyAlgorithm,           # Continuous
    "PSO": PSO,                                     # Continuous
    # Human
    "TLBO": TLBO                                    # Continuous
}
