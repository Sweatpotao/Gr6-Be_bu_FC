from collections import deque
from algorithms.base.search_base import SearchAlgorithm

class BFS(SearchAlgorithm):
    def search(self):
        start = self.problem.get_start_state()
        queue = deque([(start, [start], 0)])
        visited = set([start])

        while queue:
            state, path, cost = queue.popleft()
            self.nodes_expanded += 1

            if self.problem.is_goal(state):
                self.solution = path
                self.cost = cost
                return self._build_result()

            for next_state, step_cost in self.problem.get_neighbors(state):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path+[next_state], cost+step_cost))

        return self._build_result()
