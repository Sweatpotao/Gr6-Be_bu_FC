from abc import ABC, abstractmethod
import time

class SearchAlgorithm(ABC):
    def __init__(self, problem, config = None):
        self.problem = problem
        self.config = config or {}

        self.solution = None
        self.cost = float("inf")
        self.nodes_expanded = 0
        self.timeout = self.config.get("timeout", 300)  # Mặc định 5 phút
        self.start_time = None
        self.timed_out = False
        self.parent_map = {}  # state -> (parent_state, step_cost) for path reconstruction

    def _check_timeout(self):
        """Kiểm tra xem đã vượt quá thờii gian timeout chưa."""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            self.timed_out = True
            return True
        return False

    def _record_parent(self, child_state, parent_state, step_cost=1):
        """Record parent of a state for path reconstruction.
        
        Args:
            child_state: The child state
            parent_state: The parent state
            step_cost: Cost to move from parent to child
        """
        self.parent_map[child_state] = (parent_state, step_cost)

    def _reconstruct_path(self, start_state, goal_state):
        """Reconstruct path from start to goal using parent_map.
        
        Returns:
            tuple: (path_list, total_cost) where path_list is list of states from start to goal
        """
        if goal_state not in self.parent_map and goal_state != start_state:
            return None, 0
        
        path = [goal_state]
        total_cost = 0
        current = goal_state
        
        while current != start_state:
            if current not in self.parent_map:
                break
            parent, step_cost = self.parent_map[current]
            if parent is None:
                break
            path.append(parent)
            total_cost += step_cost
            current = parent
        
        path.reverse()
        return path, total_cost

    def _clear_parent_map(self):
        """Clear parent map to free memory."""
        self.parent_map.clear()

    @abstractmethod
    def search(self):
        """Thực hiện tìm kiếm. Phải được implement bởi lớp con."""
        pass

    #Hàm này bắt buộc con kế thừa phải cài đặt lại
    @abstractmethod
    def search(self):
        pass

    def _build_result(self):
        """Xây dựng kết quả trả về."""
        elapsed_time = 0.0
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time

        # Return clear indication when no solution is found
        if self.solution is None:
            return {
                "solution": None,
                "cost": None,
                "nodes_expanded": self.nodes_expanded,
                "found": False,
                "timeout": self.timed_out,
                "runtime": elapsed_time
            }
        return {
            "final_score": self.cost,
            "solution": self.solution,
            "nodes_expanded": self.nodes_expanded,
            "found": self.solution is not None,
            "timeout": self.timed_out,
            "runtime": elapsed_time
        }
