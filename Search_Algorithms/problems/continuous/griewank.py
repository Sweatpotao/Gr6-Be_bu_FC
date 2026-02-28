"""
Griewank Function.

Hàm đa modal với nhiều local minimum phân bố đều,
thường dùng để test khả năng tìm kiếm toàn cục.
"""

import numpy as np
from typing import Tuple
from problems.continuous.continuous_problem import ContinuousProblem


class Griewank(ContinuousProblem):
    """
    Griewank Function.
    
    f(x) = 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i)))
    
    Hàm có rất nhiều local minimum phân bố đều.
    
    Đặc điểm:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Miền tìm kiếm thường dùng: [-600, 600]^d
    - Có nhiều local minimum
    - Khó tối ưu khi số chiều cao
    
    Công thức:
        f(x) = 1 + (1/4000) * Σ(x_i²) - Π(cos(x_i / √i))
        với i = 1 đến dim
    
    Attributes:
        dim: Số chiều của không gian tìm kiếm
        bounds: Giới hạn của không gian tìm kiếm
    
    Examples:
        >>> griewank = Griewank(dim=2)
        >>> x = np.array([0.0, 0.0])
        >>> griewank.evaluate(x)
        0.0
    """

    def __init__(self, dim: int = 2, bounds: Tuple[float, float] = (-600.0, 600.0)):
        """
        Khởi tạo bài toán Griewank.

        Args:
            dim: Số chiều của không gian tìm kiếm
            bounds: Giới hạn tìm kiếm, mặc định [-600, 600]
        """
        self.dim = dim
        self.bounds = tuple(bounds)
        self.name = "Griewank"
    
    def get_dimension(self):
        return self.dim
    
    def get_bounds(self):
        return self.bounds
    
    def clone(self):
        return Griewank(self.dim, self.bounds)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Đánh giá giá trị hàm Griewank tại điểm x.

        Args:
            x: Vector điểm trong không gian tìm kiếm, shape (dim,)

        Returns:
            Giá trị f(x)

        Raises:
            ValueError: Nếu kích thước của x không khớp với dim
        """
        x = np.asarray(x)
        if x.shape[0] != self.dim:
            raise ValueError(
                f"Kích thước của x ({x.shape[0]}) không khớp với dim ({self.dim})"
            )
        
        sum_sq = np.sum(x ** 2)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        
        return float(1.0 + sum_sq / 4000.0 - prod_cos)

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Trả về điểm tối ưu toàn cục của hàm Griewank.

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
        Tính gradient của hàm Griewank tại điểm x.

        Args:
            x: Vector điểm cần tính gradient

        Returns:
            Vector gradient
        """
        x = np.asarray(x)
        
        i = np.arange(1, self.dim + 1)
        sqrt_i = np.sqrt(i)
        
        cos_term = np.cos(x / sqrt_i)
        sin_term = np.sin(x / sqrt_i)
        
        prod_cos = np.prod(cos_term)
        
        grad = np.zeros(self.dim)
        for j in range(self.dim):
            if cos_term[j] != 0:
                prod_except_j = prod_cos / cos_term[j]
                grad[j] = x[j] / 2000.0 + prod_except_j * sin_term[j] / sqrt_i[j]
            else:
                grad[j] = x[j] / 2000.0
        
        return grad


# Hàm tiện ích để tạo instance nhanh
def create_griewank(dim: int = 2, bounds: Tuple[float, float] = (-600.0, 600.0)) -> Griewank:
    """
    Tạo một instance của bài toán Griewank.

    Args:
        dim: Số chiều
        bounds: Giới hạn tìm kiếm

    Returns:
        Instance của Griewank
    """
    return Griewank(dim=dim, bounds=bounds)