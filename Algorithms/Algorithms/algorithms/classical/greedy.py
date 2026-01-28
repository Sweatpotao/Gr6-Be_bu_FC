import heapq
from algorithms.base.search_base import SearchAlgorithm

class Greedy(SearchAlgorithm):
    def search(self):
        start = self.problem.get_start_state()
        # Store: (heuristic, actual_cost, state, path)
        pq = [(self.problem.heuristic(start), 0, start, [start])]
        visited = set()

        while pq:
            _, cost, state, path = heapq.heappop(pq)
            self.nodes_expanded += 1

            if self.problem.is_goal(state):
                self.solution = path
                self.cost = cost
                return self._build_result()

            if state in visited:
                continue
            visited.add(state)

            for next_state, step_cost in self.problem.get_neighbors(state):
                if next_state not in visited:
                    h = self.problem.heuristic(next_state)
                    new_cost = cost + step_cost
                    heapq.heappush(pq, (h, new_cost, next_state, path + [next_state]))

        return self._build_result()

