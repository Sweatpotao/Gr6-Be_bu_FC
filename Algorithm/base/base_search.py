import time

class SearchResult:
    def __init__(self, success, solution, cost, nodes_expanded, time_elapsed):
        self.success = success              # True/False
        self.solution = solution            # List path [start, ..., goal]
        self.cost = cost                    # Tổng chi phí đường đi
        self.nodes_expanded = nodes_expanded # Số node đã duyệt (đánh giá hiệu năng)
        self.time_elapsed = time_elapsed    # Thời gian chạy (giây)

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILURE"
        return (f"[{status}] Cost: {self.cost}, Nodes: {self.nodes_expanded}, "
                f"Time: {self.time_elapsed:.5f}s, Path Len: {len(self.solution) if self.solution else 0}")

class SearchAlgorithm:
    def __init__(self, problem):
        self.problem = problem

    def search(self):
        raise NotImplementedError("Subclasses must implement search()")

    def reconstruct_path(self, parent_map, current_node):
        """
        Hàm tiện ích giúp truy vết ngược từ đích về start để lấy đường đi.
        """
        path = [current_node]
        while current_node in parent_map:
            current_node = parent_map[current_node]
            if current_node is None: break # Điểm bắt đầu có parent là None
            path.append(current_node)
        return path[::-1] # Đảo ngược lại