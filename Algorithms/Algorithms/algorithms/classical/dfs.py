from algorithms.base.search_base import SearchAlgorithm

class DFS(SearchAlgorithm):
    def search(self):
        start = self.problem.get_start_state()
        stack = [(start, [start], 0)]
        visited = set()

        depth_limit = self.config.get("depth_limit", float("inf"))

        while stack:
            state, path, cost = stack.pop()
            if state in visited or len(path) > depth_limit:
                continue

            visited.add(state)
            self.nodes_expanded += 1

            if self.problem.is_goal(state):
                self.solution = path
                self.cost = cost
                return self._build_result()

            for next_state, step_cost in self.problem.get_neighbors(state):
                stack.append((next_state, path+[next_state], cost+step_cost))

        return self._build_result()
