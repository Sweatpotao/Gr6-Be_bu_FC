from problems.discrete_problem import DiscreteProblem

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
                    neighbors.append(((nx, ny), 1))
        return neighbors

    def heuristic(self, state):
        # Manhattan distance
        return abs(state[0]-self.goal[0]) + abs(state[1]-self.goal[1])
