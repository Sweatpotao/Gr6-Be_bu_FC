import heapq
import time
from algorithms.base.search_base import SearchAlgorithm

class Greedy(SearchAlgorithm):
    def search(self):
        self.start_time = time.time()
        self.parent_map = {}  # Reset parent map
        
        start = self.problem.get_start_state()
        # Store: (heuristic, cost, state) - no path list
        pq = [(self.problem.heuristic(start), 0, start)]
        visited = set()
        self.parent_map[start] = (None, 0)  # start has no parent

        while pq:
            # Kiểm tra timeout
            if self._check_timeout():
                self._clear_parent_map()
                return self._build_result()

            _, cost, state = heapq.heappop(pq)
            self.nodes_expanded += 1

            if self.problem.is_goal(state):
                # Reconstruct path from parent_map
                self.solution, self.cost = self._reconstruct_path(start, state)
                self._clear_parent_map()
                return self._build_result()

            if state in visited:
                continue
            visited.add(state)

            for next_state, step_cost in self.problem.get_neighbors(state):
                if next_state not in visited:
                    h = self.problem.heuristic(next_state)
                    new_cost = cost + step_cost
                    self.parent_map[next_state] = (state, step_cost)
                    heapq.heappush(pq, (h, new_cost, next_state))

        self._clear_parent_map()
        return self._build_result()

