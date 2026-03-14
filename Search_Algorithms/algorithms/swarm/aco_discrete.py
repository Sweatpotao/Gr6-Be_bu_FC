import numpy as np
import time
from algorithms.base.optimizer_base import Optimizer


class ACO_Discrete(Optimizer):
    """
    Ant Colony Optimization for Discrete Problems.
    
    Supports multiple discrete problem types:
    - TSP: Permutation-based solution construction
    - Knapsack: Binary solution construction
    - N-Queens: Permutation-based solution
    
    Uses pheromone trails to guide solution construction with
    problem-specific heuristics.
    """

    def run(self):
        self.start_time = time.time()

        # 1. Configuration
        self.n_ants = self.config.get("n_ants", 50)
        self.max_iters = self.config.get("max_iters", 500)
        self.alpha = self.config.get("alpha", 1.0)  # Pheromone importance
        self.beta = self.config.get("beta", 2.0)    # Heuristic importance
        self.evaporation_rate = self.config.get("evaporation_rate", 0.5)
        self.Q = self.config.get("Q", 100.0)        # Pheromone deposit factor
        self.elitist_ants = self.config.get("elitist_ants", 1)  # Best ants deposit pheromone
        
        # Detect problem type
        self.problem_type = self._detect_problem_type()
        
        # Get problem dimension (handle different problem interfaces)
        if hasattr(self.problem, 'get_dimension'):
            self.n = self.problem.get_dimension()
        elif hasattr(self.problem, 'n'):
            self.n = self.problem.n
        else:
            raise AttributeError("Problem must have 'get_dimension()' method or 'n' attribute")

        # Initialize pheromone matrix
        self.pheromone = self._init_pheromone()

        # Initialize best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

        # 2. Main loop
        for iteration in range(self.max_iters):
            if self._check_timeout():
                break
            if self.evaluations >= self.max_evals:
                break

            # Build solutions for all ants
            solutions = []
            fitnesses = []

            for ant in range(self.n_ants):
                solution = self._construct_solution()
                fitness = self._evaluate_solution(solution)
                
                solutions.append(solution)
                fitnesses.append(fitness)

                # Update best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = self._copy_solution(solution)

            # Update history
            self.history.append(self.best_fitness)

            # Update pheromone trails
            self._update_pheromone(solutions, fitnesses)

        # Prepare final result
        self.solution = self.best_solution
        self.cost = self.best_fitness

        return self._build_result()

    def _detect_problem_type(self):
        """Detect the type of discrete problem."""
        problem_class = self.problem.__class__.__name__
        
        if 'TSP' in problem_class:
            return 'tsp'
        elif 'Knapsack' in problem_class:
            return 'knapsack'
        elif 'NQueens' in problem_class or 'N_Queens' in problem_class:
            return 'n_queens'
        elif 'GraphColoring' in problem_class:
            return 'graph_coloring'
        else:
            # Default to permutation-based
            return 'permutation'

    def _init_pheromone(self):
        """Initialize pheromone matrix based on problem type."""
        if self.problem_type == 'tsp':
            # Pheromone on edges: tau[i][j] = pheromone from city i to city j
            return np.ones((self.n, self.n)) * 0.1
        elif self.problem_type == 'knapsack':
            # Pheromone on items: tau[i] = pheromone for selecting item i
            return np.ones(self.n) * 0.5
        elif self.problem_type in ['n_queens', 'permutation']:
            # Pheromone on position assignments: tau[row][col]
            return np.ones((self.n, self.n)) * 0.5
        else:
            # Default: pheromone on binary decisions
            return np.ones(self.n) * 0.5

    def _construct_solution(self):
        """Construct a solution based on problem type."""
        if self.problem_type == 'tsp':
            return self._construct_tsp_solution()
        elif self.problem_type == 'knapsack':
            return self._construct_knapsack_solution()
        elif self.problem_type in ['n_queens', 'permutation']:
            return self._construct_permutation_solution()
        else:
            return self._construct_binary_solution()

    def _construct_tsp_solution(self):
        """Construct TSP tour using pheromone and distance heuristic."""
        n = self.n
        dist_matrix = self.problem.matrix
        
        # Start from random city
        tour = [np.random.randint(n)]
        unvisited = set(range(n)) - {tour[0]}

        while unvisited:
            current = tour[-1]
            
            # Calculate probabilities for unvisited cities
            probs = []
            candidates = list(unvisited)
            
            for city in candidates:
                # Pheromone from current to city
                tau = self.pheromone[current][city] ** self.alpha
                # Heuristic: inverse of distance
                dist = dist_matrix[current][city]
                if dist > 0:
                    eta = (1.0 / dist) ** self.beta
                else:
                    eta = 1.0
                probs.append(tau * eta)
            
            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(candidates)] * len(candidates)
            
            # Select next city
            next_city = np.random.choice(candidates, p=probs)
            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def _construct_knapsack_solution(self):
        """Construct Knapsack solution using pheromone and value/weight heuristic."""
        n = self.n
        solution = np.zeros(n, dtype=int)
        
        # Get items info (ratio, weight, value)
        items = self.problem.items
        capacity = self.problem.capacity
        current_weight = 0
        
        # Create candidate list based on pheromone and heuristic
        candidates = list(range(n))
        np.random.shuffle(candidates)  # Random order to avoid bias
        
        for idx in candidates:
            ratio, weight, value = items[idx]
            
            # Check if item can be added
            if current_weight + weight > capacity:
                continue
            
            # Calculate selection probability
            tau = self.pheromone[idx] ** self.alpha
            # Heuristic: value/weight ratio
            eta = (ratio if ratio != float('inf') else 1000.0) ** self.beta
            
            prob = tau * eta
            prob = prob / (1 + prob)  # Sigmoid-like normalization
            
            # Select item with probability
            if np.random.random() < prob:
                solution[idx] = 1
                current_weight += weight

        return solution

    def _construct_permutation_solution(self):
        """Construct permutation solution (for N-Queens, etc.)."""
        n = self.n
        permutation = [-1] * n
        available = set(range(n))
        
        for pos in range(n):
            if not available:
                break
                
            # Calculate probabilities for each value at this position
            probs = []
            candidates = list(available)
            
            for val in candidates:
                tau = self.pheromone[pos][val] ** self.alpha
                # Simple heuristic: prefer values that haven't been used
                eta = 1.0 ** self.beta
                probs.append(tau * eta)
            
            # Normalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(candidates)] * len(candidates)
            
            # Select value
            selected = np.random.choice(candidates, p=probs)
            permutation[pos] = selected
            available.remove(selected)

        return permutation

    def _construct_binary_solution(self):
        """Default binary solution construction."""
        n = self.n
        solution = np.zeros(n, dtype=int)
        
        for i in range(n):
            # Use pheromone as probability
            prob = self.pheromone[i] ** self.alpha
            prob = prob / (1 + prob)
            solution[i] = 1 if np.random.random() < prob else 0

        return solution

    def _evaluate_solution(self, solution):
        """Evaluate a solution based on problem type."""
        if self.problem_type == 'tsp':
            return self._evaluate_tsp(solution)
        elif self.problem_type == 'knapsack':
            return self._evaluate_knapsack(solution)
        else:
            # Use problem's evaluate method if available
            if hasattr(self.problem, 'evaluate'):
                result = self.evaluate(solution)
                if result is not None:
                    return result
            # Fallback: try to convert solution and evaluate
            return self._evaluate_generic(solution)

    def _evaluate_tsp(self, tour):
        """Evaluate TSP tour cost."""
        dist_matrix = self.problem.matrix
        cost = 0
        n = len(tour)
        
        for i in range(n - 1):
            cost += dist_matrix[tour[i]][tour[i + 1]]
        # Return to start
        cost += dist_matrix[tour[-1]][tour[0]]
        
        return cost

    def _evaluate_knapsack(self, solution):
        """Evaluate Knapsack solution with penalty for overweight."""
        items = self.problem.items
        capacity = self.problem.capacity
        
        total_weight = 0
        total_value = 0
        
        for i, selected in enumerate(solution):
            if selected == 1:
                _, weight, value = items[i]
                total_weight += weight
                total_value += value
        
        # Penalty for exceeding capacity
        if total_weight > capacity:
            penalty = 1000 * (total_weight - capacity)
            return -total_value + penalty
        
        # Return negative value (minimization problem)
        return -total_value

    def _evaluate_generic(self, solution):
        """Generic evaluation using problem's evaluate method."""
        try:
            result = self.evaluate(solution)
            return result if result is not None else float('inf')
        except:
            return float('inf')

    def _update_pheromone(self, solutions, fitnesses):
        """Update pheromone trails."""
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Find best solutions for elitist update
        sorted_indices = np.argsort(fitnesses)
        
        # Update based on best ants (elitist strategy)
        for rank, idx in enumerate(sorted_indices[:self.elitist_ants]):
            solution = solutions[idx]
            fitness = fitnesses[idx]
            
            if fitness == float('inf'):
                continue
            
            # Deposit pheromone
            deposit = self.Q / (1 + fitness)
            self._deposit_pheromone(solution, deposit)

    def _deposit_pheromone(self, solution, amount):
        """Deposit pheromone based on solution type."""
        if self.problem_type == 'tsp':
            # Deposit on tour edges
            n = len(solution)
            for i in range(n - 1):
                self.pheromone[solution[i]][solution[i + 1]] += amount
            # Close the tour
            self.pheromone[solution[-1]][solution[0]] += amount
        elif self.problem_type == 'knapsack':
            # Deposit on selected items
            for i, selected in enumerate(solution):
                if selected == 1:
                    self.pheromone[i] += amount
        elif self.problem_type in ['n_queens', 'permutation']:
            # Deposit on position assignments
            for pos, val in enumerate(solution):
                if 0 <= val < self.n:
                    self.pheromone[pos][val] += amount
        else:
            # Default binary
            for i, val in enumerate(solution):
                if val == 1:
                    self.pheromone[i] += amount

    def _copy_solution(self, solution):
        """Create a copy of the solution."""
        if isinstance(solution, list):
            return solution.copy()
        elif isinstance(solution, np.ndarray):
            return solution.copy()
        return solution

    def _check_timeout(self):
        """Check if timeout has been exceeded."""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            self.timed_out = True
            return True
        return False
