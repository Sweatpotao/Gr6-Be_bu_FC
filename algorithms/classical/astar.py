import heapq
from algorithms.base.search_base import SearchAlgorithm

class AStar(SearchAlgorithm):
    def search(self):
        start = self.problem.get_start_state()
        pq = [(self.problem.heuristic(start), 0, start, [start])]
        visited = {}

        while pq:
            f, g, state, path = heapq.heappop(pq)
            self.nodes_expanded += 1

            if state in visited and visited[state] <= g:
                continue
            visited[state] = g

            if self.problem.is_goal(state):
                self.solution = path
                self.cost = g
                return self._build_result()

            for next_state, step_cost in self.problem.get_neighbors(state):
                new_g = g + step_cost
                h = self.problem.heuristic(next_state)
                heapq.heappush(
                    pq,
                    (new_g + h, new_g, next_state, path + [next_state])
                )

        return self._build_result()
