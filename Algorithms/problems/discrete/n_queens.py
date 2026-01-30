from problems.discrete.discrete_problem import DiscreteProblem

class NQueens(DiscreteProblem):
    """
    N-Queens Problem.
    
    Place N queens on an N×N chessboard such that no two queens attack each other.
    A state is represented as a tuple where index represents column and value represents row.
    """
    
    def __init__(self, n: int = 8):
        self.n = n
    
    def get_start_state(self):
        # Start with all queens in row 0 (same row)
        return tuple([0 for _ in range(self.n)])
    
    def is_goal(self, state):
        if len(state) != self.n:
            return False
        
        # Check for row conflicts
        if len(set(state)) != self.n:
            return False
        
        # Check for diagonal conflicts
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(i - j) == abs(state[i] - state[j]):
                    return False
        
        return True
    
    def get_neighbors(self, state):
        neighbors = []
        
        for col in range(self.n):
            current_row = state[col]
            
            for new_row in range(self.n):
                if new_row != current_row:
                    # Create new state by moving queen in column 'col' to row 'new_row'
                    new_state = list(state)
                    new_state[col] = new_row
                    new_state = tuple(new_state)
                    neighbors.append((new_state, 1))
        
        return neighbors
    
    def heuristic(self, state):
        """
        Heuristic function: count number of conflicting pairs of queens.
        0 means goal state.
        """
        conflicts = 0
        
        # Check row conflicts
        row_counts = {}
        for row in state:
            row_counts[row] = row_counts.get(row, 0) + 1
        for count in row_counts.values():
            conflicts += count * (count - 1) // 2
        
        # Check diagonal conflicts
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(i - j) == abs(state[i] - state[j]):
                    conflicts += 1
        
        return conflicts
    
    def clone(self):
        return NQueens(self.n)
