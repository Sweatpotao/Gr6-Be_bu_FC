import heapq
import time
from algorithms.base.search_base import SearchAlgorithm

class AStar(SearchAlgorithm):
    def search(self):
        self.start_time = time.time()
        self.parent_map = {}  # Reset parent map
        
        start = self.problem.get_start_state()
        start_h = self.problem.heuristic(start)
        pq = [(start_h, 0, start)]  # (f, g, state) - no path list
        visited = {}
        self.parent_map[start] = (None, 0)  # start has no parent

        while pq:
            # Kiểm tra timeout
            if self._check_timeout():
                self._clear_parent_map()
                return self._build_result()

            f, g, state = heapq.heappop(pq)
            self.nodes_expanded += 1

            if state in visited and visited[state] <= g:
                continue
            visited[state] = g

            if self.problem.is_goal(state):
                # Reconstruct path from parent_map
                self.solution, self.cost = self._reconstruct_path(start, state)
                self._clear_parent_map()
                return self._build_result()

            for next_state, step_cost in self.problem.get_neighbors(state):
                new_g = g + step_cost
                # Only process if we found a better path
                if next_state not in visited or visited[next_state] > new_g:
                    h = self.problem.heuristic(next_state)
                    self.parent_map[next_state] = (state, step_cost)
                    heapq.heappush(pq, (new_g + h, new_g, next_state))

        self._clear_parent_map()
        return self._build_result()
