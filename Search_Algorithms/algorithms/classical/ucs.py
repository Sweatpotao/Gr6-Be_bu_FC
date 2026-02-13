import heapq
import time
from algorithms.base.search_base import SearchAlgorithm

class UCS(SearchAlgorithm):
    def search(self):
        self.start_time = time.time()
        self.parent_map = {}  # Reset parent map
        
        start = self.problem.get_start_state()
        pq = [(0, start)]  # (cost, state) - no path list
        visited = {}
        self.parent_map[start] = (None, 0)  # start has no parent

        while pq:
            # Kiểm tra timeout
            if self._check_timeout():
                self._clear_parent_map()
                return self._build_result()

            cost, state = heapq.heappop(pq)
            self.nodes_expanded += 1

            if state in visited and visited[state] <= cost:
                continue
            visited[state] = cost

            if self.problem.is_goal(state):
                # Reconstruct path from parent_map
                self.solution, self.cost = self._reconstruct_path(start, state)
                self._clear_parent_map()
                return self._build_result()

            for next_state, step_cost in self.problem.get_neighbors(state):
                new_cost = cost + step_cost
                # Only process if we found a better path
                if next_state not in visited or visited[next_state] > new_cost:
                    self.parent_map[next_state] = (state, step_cost)
                    heapq.heappush(pq, (new_cost, next_state))

        self._clear_parent_map()
        return self._build_result()

