import time
import heapq
from ..base.base_search import SearchAlgorithm, SearchResult

class GreedySearch(SearchAlgorithm):
    def search(self):
        start_time = time.time()
        start = self.problem.get_start_state()
        
        # Priority Queue: (heuristic, current_node)
        frontier = [(self.problem.heuristic(start), start)]
        parent_map = {start: None}
        explored = {start}
        nodes_expanded = 0

        while frontier:
            _, current = heapq.heappop(frontier)
            nodes_expanded += 1

            if self.problem.is_goal(current):
                path = self.reconstruct_path(parent_map, current)
                # Tính cost thực tế sau khi tìm xong
                cost = len(path) - 1 
                return SearchResult(True, path, cost, nodes_expanded, time.time() - start_time)

            for neighbor in self.problem.get_neighbors(current):
                if neighbor not in explored:
                    explored.add(neighbor)
                    parent_map[neighbor] = current
                    priority = self.problem.heuristic(neighbor)
                    heapq.heappush(frontier, (priority, neighbor))

        return SearchResult(False, [], 0, nodes_expanded, time.time() - start_time)