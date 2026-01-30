from problems.continuous.sphere import SphereFunction
from problems.discrete.grid_pathfinding import GridPathfinding

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
    "grid_pathfinding": GridPathfinding
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
