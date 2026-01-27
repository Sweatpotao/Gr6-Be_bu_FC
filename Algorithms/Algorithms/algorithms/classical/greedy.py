import heapq
from algorithms.base.search_base import SearchAlgorithm

class Greedy(SearchAlgorithm):
    def search(self):
        start = self.problem.get_start_state()
        pq = [(self.problem.heuristic(start), start, [start])]
        visited = set()

        while pq:
            _, state, path = heapq.heappop(pq)
            self.nodes_expanded += 1

            if self.problem.is_goal(state):
                self.solution = path
                self.cost = len(path) - 1
                return self._build_result()

            if state in visited:
                continue
            visited.add(state)

            for next_state, step_cost in self.problem.get_neighbors(state):
                h = self.problem.heuristic(next_state)
                heapq.heappush(pq, (h, next_state, path + [next_state]))

        return self._build_result()

