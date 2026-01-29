from problems.discrete_problem import DiscreteProblem
import math

class GridPathfinding(DiscreteProblem):
    def __init__(self, grid, start, goal, diagonal=False):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.diagonal = diagonal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def get_start_state(self):
        return self.start

    def is_goal(self, state):
        return state == self.goal

    def get_neighbors(self, state):
        x, y = state
        moves = [(0,1),(1,0),(0,-1),(-1,0)]
        if self.diagonal:
            moves += [(1,1),(1,-1),(-1,1),(-1,-1)]

        neighbors = []
        for dx, dy in moves:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid[nx][ny] == 0:
                    # Diagonal moves cost sqrt(2), orthogonal cost 1
                    step_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1
                    neighbors.append(((nx, ny), step_cost))
        return neighbors

    def heuristic(self, state):
        if self.diagonal:
            # Octile distance - admissible for 8-directional grid with diagonal moves
            dx = abs(state[0] - self.goal[0])
            dy = abs(state[1] - self.goal[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            # Manhattan distance for 4-directional grid
            return abs(state[0]-self.goal[0]) + abs(state[1]-self.goal[1])

    def clone(self):
        """Create a fresh copy of this problem for independent runs."""
        return GridPathfinding(
            grid=[row[:] for row in self.grid],  # Deep copy grid
            start=self.start,
            goal=self.goal,
            diagonal=self.diagonal
        )
