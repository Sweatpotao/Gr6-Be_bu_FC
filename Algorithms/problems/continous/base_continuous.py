"""
Lớp cơ sở cho các bài toán tối ưu liên tục (continuous optimization problems).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class ContinuousProblem(ABC):
    """
    Lớp cơ sở cho các bài toán tối ưu liên tục.
    
    Các bài toán tối ưu liên tục tìm kiếm vector x trong không gian liên tục
    để tối thiểu hóa (hoặc tối đa hóa) một hàm mục tiêu f(x).
    
    Attributes:
        dim: Số chiều của không gian tìm kiếm
        bounds: Giới hạn của không gian tìm kiếm, dạng [(min, max)] * dim
        name: Tên của bài toán
    """

    def __init__(self, dim: int = 2, bounds: Optional[List[Tuple[float, float]]] = None, name: str = "ContinuousProblem"):
        """
        Khởi tạo bài toán tối ưu liên tục.

        Args:
            dim: Số chiều của không gian tìm kiếm
            bounds: Giới hạn của không gian tìm kiếm, list của (min, max) tuples
            name: Tên của bài toán
        """
        self.dim = dim
        self.name = name
        
        if bounds is None:
            # Mặc định giới hạn [-5.12, 5.12] cho mỗi chiều
            self.bounds = [(-5.12, 5.12)] * dim
        else:
            self.bounds = bounds

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """
        Đánh giá giá trị hàm mục tiêu tại điểm x.

        Args:
            x: Vector điểm trong không gian tìm kiếm, shape (dim,)

        Returns:
            Giá trị hàm mục tiêu (càng nhỏ càng tốt cho bài toán minimization)
        """
        pass

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Trả về điểm tối ưu toàn cục và giá trị hàm mục tiêu tại đó.

        Returns:
            Tuple của (x_opt, f_opt) trong đó:
            - x_opt: Vector điểm tối ưu toàn cục
            - f_opt: Giá trị hàm mục tiêu tại điểm tối ưu
        """
        x_opt = np.zeros(self.dim)
        f_opt = 0.0
        return x_opt, f_opt

    def is_valid(self, x: np.ndarray) -> bool:
        """
        Kiểm tra xem điểm x có nằm trong miền hợp lệ không.

        Args:
            x: Vector điểm cần kiểm tra

        Returns:
            True nếu x nằm trong bounds, False nếu không
        """
        if len(x) != self.dim:
            return False
        
        for i, (xi, (lower, upper)) in enumerate(zip(x, self.bounds)):
            if xi < lower or xi > upper:
                return False
        return True

    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Tạo một giải pháp ngẫu nhiên trong không gian tìm kiếm.

        Args:
            rng: Random number generator (tùy chọn)

        Returns:
            Vector ngẫu nhiên trong bounds
        """
        if rng is None:
            rng = np.random.default_rng()
        
        x = np.zeros(self.dim)
        for i, (lower, upper) in enumerate(self.bounds):
            x[i] = rng.uniform(lower, upper)
        return x

    def __repr__(self) -> str:
        return f"{self.name}(dim={self.dim})"
