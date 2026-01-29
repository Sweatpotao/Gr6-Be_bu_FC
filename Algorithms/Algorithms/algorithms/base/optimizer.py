from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Optimizer(ABC):
    """
    Lớp cha (base class) cho tất cả các thuật toán tối ưu.
    
    Lớp này định nghĩa interface chung cho mọi thuật toán tối ưu,
    bao gồm cả thuật toán tìm kiếm (search) và thuật toán tối ưu hóa (optimization).
    
    Attributes:
        problem: Bài toán cần tối ưu (thuộc về lớp BaseProblem)
        config: Cấu hình tham số cho thuật toán (dict hoặc None)
        best_solution: Giải pháp tốt nhất tìm được
        best_fitness: Giá trị fitness/objective tốt nhất (đối với tối ưu liên tục)
        cost: Chi phí của giải pháp (đối với bài toán tìm kiếm)
        iterations: Số lần lặp/lặp lại của thuật toán
        evaluations: Số lần đánh giá hàm mục tiêu
    """

    def __init__(self, problem: Any, config: Optional[Dict] = None):
        """
        Khởi tạo Optimizer.

        Args:
            problem: Bài toán cần giải quyết
            config: Dictionary chứa các tham số cấu hình cho thuật toán
        """
        self.problem = problem
        self.config = config if config is not None else {}

        # Kết quả tối ưu
        self.best_solution: Any = None
        self.best_fitness: float = float("inf")
        self.cost: float = float("inf")

        # Thống kê thuật toán
        self.iterations: int = 0
        self.evaluations: int = 0

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """
        Phương thức trừu tượng - thực hiện quá trình tối ưu.
        
        Mọi lớp con kế thừa từ Optimizer PHẢI cài đặt lại phương thức này.

        Returns:
            Dictionary chứa kết quả tối ưu với các key:
            - 'solution': Giải pháp tìm được
            - 'best_fitness' hoặc 'cost': Giá trị tối ưu
            - 'iterations': Số lần lặp
            - 'evaluations': Số lần đánh giá
            - 'found' hoặc 'success': Trạng thái tìm kiếm/tối ưu thành công
        """
        pass

    def _evaluate(self, solution: Any) -> float:
        """
        Đánh giá một giải pháp.
        
        Phương thức này có thể được override trong lớp con.
        Mặc định gọi phương thức evaluate của problem nếu có.

        Args:
            solution: Giải pháp cần đánh giá

        Returns:
            Giá trị fitness/objective (float), càng nhỏ càng tốt (minimization)
        """
        self.evaluations += 1
        if hasattr(self.problem, 'evaluate'):
            return self.problem.evaluate(solution)
        return float("inf")

    def _build_result(self, **kwargs) -> Dict[str, Any]:
        """
        Xây dựng dictionary kết quả chuẩn.

        Args:
            **kwargs: Các key-value bổ sung cho kết quả

        Returns:
            Dictionary kết quả tối ưu
        """
        result = {
            "solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "cost": self.cost,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "found": self.best_solution is not None
        }
        result.update(kwargs)
        return result

    def reset(self) -> None:
        """
        Reset trạng thái của optimizer về ban đầu.
        
        Hữu ích khi muốn chạy lại thuật toán với cùng problem/config.
        """
        self.best_solution = None
        self.best_fitness = float("inf")
        self.cost = float("inf")
        self.iterations = 0
        self.evaluations = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiện tại của optimizer.

        Returns:
            Dictionary chứa thống kê thuật toán
        """
        return {
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "best_fitness": self.best_fitness,
            "cost": self.cost
        }
