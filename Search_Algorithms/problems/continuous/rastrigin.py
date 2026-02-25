"""
Rastrigin Function.

Hàm đa modal với rất nhiều local minimum, thường dùng để test khả năng
thoát khỏi local minimum của thuật toán tối ưu.
"""

import numpy as np
from typing import Tuple
from problems.continuous.continuous_problem import ContinuousProblem


class Rastrigin(ContinuousProblem):
    """
    Rastrigin Function: f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Hàm đa modal phức tạp với rất nhiều local minimum đều đặn phân bố.
    
    Đặc điểm:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Miền tìm kiếm thường dùng: [-5.12, 5.12]^d
    - Có khoảng 10^d local minimum
    - Rất khó để tối ưu do có nhiều cực trị địa phương
    
    Công thức:
        f(x) = 10d + Σ(x_i² - 10*cos(2π*x_i))
        với i = 1 đến dim
    
    Attributes:
        dim: Số chiều của không gian tìm kiếm
        bounds: Giới hạn của không gian tìm kiếm
        A: Hệ số điều chỉnh độ sâu của các local minimum (mặc định 10)
    
    Examples:
        >>> rastrigin = Rastrigin(dim=2)
        >>> x = np.array([0.0, 0.0])
        >>> rastrigin.evaluate(x)
        0.0
    """

    def __init__(self, dim: int = 2, bounds: Tuple[float, float] = (-5.12, 5.12), A: float = 10.0):
        """
        Khởi tạo bài toán Rastrigin.

        Args:
            dim: Số chiều của không gian tìm kiếm
            bounds: Giới hạn tìm kiếm, mặc định [-5.12, 5.12]
            A: Hệ số điều chỉnh, mặc định 10
        """
        self.dim = dim
        self.bounds = tuple(bounds)
        self.name = "Rastrigin"
        self.A = A
    
    def get_dimension(self):
        return self.dim
    
    def get_bounds(self):
        return self.bounds
    
    def clone(self):
        return Rastrigin(self.dim, self.bounds, self.A)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Đánh giá giá trị hàm Rastrigin tại điểm x.

        Args:
            x: Vector điểm trong không gian tìm kiếm, shape (dim,)

        Returns:
            Giá trị f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i))

        Raises:
            ValueError: Nếu kích thước của x không khớp với dim
        """
        x = np.asarray(x)
        if x.shape[0] != self.dim:
            raise ValueError(
                f"Kích thước của x ({x.shape[0]}) không khớp với dim ({self.dim})"
            )
        
        return float(self.A * self.dim + np.sum(x ** 2 - self.A * np.cos(2 * np.pi * x)))

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Trả về điểm tối ưu toàn cục của hàm Rastrigin.

        Returns:
            Tuple (x_opt, f_opt) trong đó:
            - x_opt: Vector [0, 0, ..., 0]
            - f_opt: 0.0
        """
        x_opt = np.zeros(self.dim)
        f_opt = 0.0
        return x_opt, f_opt

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Tính gradient của hàm Rastrigin tại điểm x.

        Args:
            x: Vector điểm cần tính gradient

        Returns:
            Vector gradient: grad(x)_i = 2*x_i + 20*pi*sin(2*pi*x_i)
        """
        x = np.asarray(x)
        return 2.0 * x + 2.0 * self.A * np.pi * np.sin(2.0 * np.pi * x)


# Hàm tiện ích để tạo instance nhanh
def create_rastrigin(dim: int = 2, bounds: Tuple[float, float] = (-5.12, 5.12), A: float = 10.0) -> Rastrigin:
    """
    Tạo một instance của bài toán Rastrigin.

    Args:
        dim: Số chiều
        bounds: Giới hạn tìm kiếm
        A: Hệ số điều chỉnh

    Returns:
        Instance của Rastrigin
    """
    return Rastrigin(dim=dim, bounds=bounds, A=A)