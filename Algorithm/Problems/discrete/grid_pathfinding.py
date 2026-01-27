import numpy as np
from ..base_problem import Problem

class GridPathFinding(Problem):
    def __init__(self, grid_matrix, start, goal, heuristic_type="manhattan"):
        """
        grid_matrix: Ma trận 2D (0: đường đi, 1: vật cản)
        """
        self.grid = np.array(grid_matrix)
        self.rows, self.cols = self.grid.shape
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.heuristic_type = heuristic_type

    def get_start_state(self):
        return self.start

    def is_goal(self, state):
        return state == self.goal

    def get_neighbors(self, state):
        r, c = state
        neighbors = []
        # 4 hướng: Lên, Xuống, Trái, Phải
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Kiểm tra biên và vật cản (0 là free, khác 0 là blocked)
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] == 0:
                    neighbors.append((nr, nc))
        return neighbors

    def get_cost(self, state, neighbor):
        return 1 # Cost đều bằng 1 trên lưới ô vuông

    def heuristic(self, state):
        """Tính khoảng cách ước lượng đến đích"""
        x1, y1 = state
        x2, y2 = self.goal
        
        if self.heuristic_type == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        elif self.heuristic_type == "euclidean":
            return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        return 0