from abc import ABC, abstractmethod

class Problem(ABC):
    """
    Abstract Base Class cho tất cả các bài toán.
    Định nghĩa các phương thức mà mọi bài toán (Grid, TSP, N-Queens...) phải có.
    """

    # --- Methods for Search Problems (Discrete) ---
    @abstractmethod
    def get_start_state(self):
        """Trả về trạng thái bắt đầu"""
        pass

    @abstractmethod
    def is_goal(self, state):
        """Kiểm tra xem trạng thái hiện tại có phải đích không"""
        pass

    @abstractmethod
    def get_neighbors(self, state):
        """Trả về danh sách các trạng thái lân cận hợp lệ"""
        pass

    def get_cost(self, state, neighbor):
        """
        Trả về chi phí di chuyển từ state sang neighbor.
        Mặc định là 1 (cho BFS/DFS). Các bài toán có trọng số cần override hàm này.
        """
        return 1
    
    def heuristic(self, state):
        """
        Hàm heuristic dùng cho A* và Greedy.
        Mặc định trả về 1 (tương đương Dijkstra/UCS).
        """
        return 1