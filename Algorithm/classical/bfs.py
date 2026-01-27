import time
from collections import deque
from ..base.base_search import SearchAlgorithm, SearchResult

class BFS(SearchAlgorithm):
    def search(self):
        start_time = time.time()
        start = self.problem.get_start_state()
        
        if self.problem.is_goal(start):
            return SearchResult(True, [start], 0, 0, 0)

        frontier = deque([start])   # Queue (FIFO)
        parent_map = {start: None}  # Lưu vết: con -> cha
        explored = {start}
        nodes_expanded = 0

        while frontier:
            current = frontier.popleft()
            nodes_expanded += 1

            if self.problem.is_goal(current):
                path = self.reconstruct_path(parent_map, current)
                return SearchResult(True, path, len(path)-1, nodes_expanded, time.time() - start_time)

            for neighbor in self.problem.get_neighbors(current):
                if neighbor not in explored:
                    explored.add(neighbor)
                    parent_map[neighbor] = current
                    frontier.append(neighbor)

        return SearchResult(False, [], 0, nodes_expanded, time.time() - start_time)