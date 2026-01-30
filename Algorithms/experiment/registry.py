from problems.continuous.sphere import SphereFunction
from problems.continuous.ackley import Ackley
from problems.continuous.rastrigin import Rastrigin
from problems.continuous.rosenbrock import Rosenbrock
from problems.continuous.griewnak import Griewank
from problems.discrete.grid_pathfinding import GridPathfinding
from problems.discrete.n_queens import NQueens

from algorithms.local_search.hill_climbing import HillClimbing
from algorithms.local_search.simulated_annealing import SimulatedAnnealing
#from algorithms.swarm.pso import PSO
from algorithms.classical.bfs import BFS
from algorithms.classical.astar import AStar
from algorithms.classical.dfs import DFS
from algorithms.classical.ucs import UCS
from algorithms.classical.greedy import Greedy

PROBLEM_REGISTRY = {
    "sphere": SphereFunction,
    "ackley": Ackley,
    "rastrigin": Rastrigin,
    "rosenbrock": Rosenbrock,
    "griewank": Griewank,
    "grid_pathfinding": GridPathfinding,
    "n_queens": NQueens
}

ALGORITHM_REGISTRY = {
    "HillClimbing": HillClimbing,
    "SimulatedAnnealing": SimulatedAnnealing,
    "BFS": BFS,
    "DFS": DFS,
    "UCS": UCS,
    "Greedy": Greedy,
    "AStar": AStar,
}
