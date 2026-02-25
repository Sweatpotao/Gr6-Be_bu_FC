import time
from algorithms.base.search_base import SearchAlgorithm

class DFS(SearchAlgorithm):
    def search(self):
        self.start_time = time.time()
        self.parent_map = {}  # Reset parent map
        
        start = self.problem.get_start_state()
        stack = [(start, 0)]  # (state, cost) - no path list
        visited = set([start])
        self.parent_map[start] = (None, 0)  # start has no parent
        
        # Track depth separately for depth limit
        depth_map = {start: 1}  # state -> depth
        
        depth_limit = self.config.get("depth_limit", float("inf"))

        while stack:
            # Kiểm tra timeout
            if self._check_timeout():
                self._clear_parent_map()
                return self._build_result()

            state, cost = stack.pop()
            current_depth = depth_map.get(state, 1)
            
            # Skip if exceeds depth limit
            if current_depth > depth_limit:
                continue

            self.nodes_expanded += 1

            if self.problem.is_goal(state):
                # Reconstruct path from parent_map
                self.solution, self.cost = self._reconstruct_path(start, state)
                self._clear_parent_map()
                return self._build_result()

            # Only add neighbors that haven't been visited and within depth limit
            for next_state, step_cost in self.problem.get_neighbors(state):
                if next_state not in visited and current_depth + 1 <= depth_limit:
                    visited.add(next_state)
                    self.parent_map[next_state] = (state, step_cost)
                    depth_map[next_state] = current_depth + 1
                    stack.append((next_state, cost + step_cost))

        self._clear_parent_map()
        return self._build_result()
