import heapq
from algorithms.base.search_base import SearchAlgorithm

class UCS(SearchAlgorithm):
    def search(self):
        start = self.problem.get_start_state()
        pq = [(0, start, [start])]
        visited = {}

        while pq:
            cost, state, path = heapq.heappop(pq)
            self.nodes_expanded += 1

            if state in visited and visited[state] <= cost:
                continue
            visited[state] = cost

            if self.problem.is_goal(state):
                self.solution = path
                self.cost = cost
                return self._build_result()

            for next_state, step_cost in self.problem.get_neighbors(state):
                heapq.heappush(
                    pq,
                    (cost + step_cost, next_state, path + [next_state])
                )

        return self._build_result()

