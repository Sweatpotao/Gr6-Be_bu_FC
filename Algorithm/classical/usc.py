import time
import heapq
from ..base.base_search import SearchAlgorithm, SearchResult

class UCS(SearchAlgorithm):
    def search(self):
        start_time = time.time()
        start = self.problem.get_start_state()
        
        # Priority Queue: (cost_g, current_node)
        frontier = [(0, start)]
        parent_map = {start: None}
        cost_so_far = {start: 0} # g(n)
        nodes_expanded = 0

        while frontier:
            current_cost, current = heapq.heappop(frontier)
            nodes_expanded += 1

            if self.problem.is_goal(current):
                path = self.reconstruct_path(parent_map, current)
                return SearchResult(True, path, current_cost, nodes_expanded, time.time() - start_time)

            for neighbor in self.problem.get_neighbors(current):
                new_cost = current_cost + self.problem.get_cost(current, neighbor)
                
                # Nếu chưa đi qua hoặc tìm được đường rẻ hơn
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    parent_map[neighbor] = current
                    heapq.heappush(frontier, (new_cost, neighbor))

        return SearchResult(False, [], 0, nodes_expanded, time.time() - start_time)