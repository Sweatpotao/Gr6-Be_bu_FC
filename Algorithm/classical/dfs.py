import time
from ..base.base_search import SearchAlgorithm, SearchResult

class DFS(SearchAlgorithm):
    def search(self):
        start_time = time.time()
        start = self.problem.get_start_state()
        
        frontier = [start]          # Stack (LIFO)
        parent_map = {start: None}
        explored = set()            # Chỉ add vào explored khi đã POP ra khỏi stack
        nodes_expanded = 0

        while frontier:
            current = frontier.pop()
            
            if current in explored:
                continue
            
            explored.add(current)
            nodes_expanded += 1

            if self.problem.is_goal(current):
                path = self.reconstruct_path(parent_map, current)
                return SearchResult(True, path, len(path)-1, nodes_expanded, time.time() - start_time)

            # Đảo ngược neighbors để DFS duyệt theo thứ tự tự nhiên (tùy chọn)
            for neighbor in reversed(self.problem.get_neighbors(current)):
                if neighbor not in explored: # Kiểm tra sơ bộ để giảm kích thước stack
                    # Lưu ý: DFS graph search cập nhật parent hơi phức tạp vì 1 node có thể add nhiều lần
                    # Cách đơn giản: Chỉ ghi nhận parent lần đầu thấy
                    if neighbor not in parent_map:
                         parent_map[neighbor] = current
                    frontier.append(neighbor)

        return SearchResult(False, [], 0, nodes_expanded, time.time() - start_time)