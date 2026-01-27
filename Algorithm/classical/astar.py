import time
import heapq
from ..base.base_search import SearchAlgorithm, SearchResult

class AStar(SearchAlgorithm):
    def search(self):
        start_time = time.time()
        start = self.problem.get_start_state()
        
        # Priority Queue: (f_score, current_node)
        # f = g + h
        start_g = 0
        start_h = self.problem.heuristic(start)
        frontier = [(start_g + start_h, start)]
        
        parent_map = {start: None}
        cost_so_far = {start: 0} # g(n)
        nodes_expanded = 0

        while frontier:
            _, current = heapq.heappop(frontier)
            nodes_expanded += 1

            if self.problem.is_goal(current):
                path = self.reconstruct_path(parent_map, current)
                return SearchResult(True, path, cost_so_far[current], nodes_expanded, time.time() - start_time)

            for neighbor in self.problem.get_neighbors(current):
                new_cost = cost_so_far[current] + self.problem.get_cost(current, neighbor)
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    parent_map[neighbor] = current
                    
                    priority = new_cost + self.problem.heuristic(neighbor)
                    heapq.heappush(frontier, (priority, neighbor))

        return SearchResult(False, [], 0, nodes_expanded, time.time() - start_time)