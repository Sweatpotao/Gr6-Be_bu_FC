from collections import deque
import time
from algorithms.base.search_base import SearchAlgorithm

class BFS(SearchAlgorithm):
    def search(self):
        self.start_time = time.time()
        self.parent_map = {}  # Reset parent map
        
        start = self.problem.get_start_state()
        queue = deque([(start, 0)])  # (state, cost) - no path list
        visited = set([start])
        self.parent_map[start] = (None, 0)  # start has no parent

        while queue:
            # Kiểm tra timeout
            if self._check_timeout():
                self._clear_parent_map()
                return self._build_result()

            state, cost = queue.popleft()
            self.nodes_expanded += 1

            if self.problem.is_goal(state):
                # Reconstruct path from parent_map
                self.solution, self.cost = self._reconstruct_path(start, state)
                self._clear_parent_map()
                return self._build_result()

            for next_state, step_cost in self.problem.get_neighbors(state):
                if next_state not in visited:
                    visited.add(next_state)
                    self.parent_map[next_state] = (state, step_cost)
                    queue.append((next_state, cost + step_cost))

        self._clear_parent_map()
        return self._build_result()
