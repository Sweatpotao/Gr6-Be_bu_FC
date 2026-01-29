"""
Griewank Function.

Hàm có rất nhiều local minimum phân bố đều với giá trị rất gần global minimum.
Thường dùng để test khả năng phân biệt local và global minimum.
"""

import numpy as np
from typing import List, Tuple, Optional
from problems.continous.base_continuous import ContinuousProblem


class Griewank(ContinuousProblem):
    """
    Griewank Function.
    
    f(x) = 1 + 1/4000 * sum(x_i^2) - prod(cos(x_i / sqrt(i)))
    
    Hàm có rất nhiều local minimum với giá trị rất gần global minimum.
    Khi số chiều tăng, số local minimum tăng rất nhanh.
    
    Đặc điểm:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Miền tìm kiếm thường dùng: [-600, 600]^d
    - Có rất nhiều local minimum phân bố đều
    - Các local minimum có giá trị rất gần global minimum (khó phân biệt)
    
    Công thức:
        f(x) = 1 + (1/4000) * Σx_i² - Πcos(x_i / sqrt(i))
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

    def __init__(self, dim: int = 2, bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Khởi tạo bài toán Griewank.

        Args:
            dim: Số chiều của không gian tìm kiếm
            bounds: Giới hạn tìm kiếm, mặc định [-600, 600]^dim
        """
        if bounds is None:
            bounds = [(-600.0, 600.0)] * dim
        
        super().__init__(dim=dim, bounds=bounds, name="Griewank")

    def evaluate(self, x: np.ndarray) -> float:
        """
        Đánh giá giá trị hàm Griewank tại điểm x.

        Args:
            x: Vector điểm trong không gian tìm kiếm, shape (dim,)

        Returns:
            Giá trị f(x) = 1 + 1/4000*sum(x_i^2) - prod(cos(x_i/sqrt(i)))

        Raises:
            ValueError: Nếu kích thước của x không khớp với dim
        """
        x = np.asarray(x)
        if x.shape[0] != self.dim:
            raise ValueError(f"Kích thước của x ({x.shape[0]}) không khớp với dim ({self.dim})")
        
        # Tính sum term
        sum_term = np.sum(x ** 2) / 4000.0
        
        # Tính product term: product of cos(x_i / sqrt(i))
        # Lưu ý: i bắt đầu từ 1 (không phải 0)
        prod_term = 1.0
        for i in range(self.dim):
            prod_term *= np.cos(x[i] / np.sqrt(i + 1))
        
        return float(1.0 + sum_term - prod_term)

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
        grad = np.zeros(self.dim)
        
        # Precompute cosines and sines
        cos_vals = np.zeros(self.dim)
        sin_vals = np.zeros(self.dim)
        for i in range(self.dim):
            cos_vals[i] = np.cos(x[i] / np.sqrt(i + 1))
            sin_vals[i] = np.sin(x[i] / np.sqrt(i + 1))
        
        # Precompute product terms (product of all cos except i)
        total_prod = np.prod(cos_vals)
        
        for i in range(self.dim):
            # Gradient from sum term: (2*x_i) / 4000
            grad[i] = x[i] / 2000.0
            
            # Gradient from product term
            if cos_vals[i] != 0:
                prod_except_i = total_prod / cos_vals[i]
                grad[i] += prod_except_i * sin_vals[i] / np.sqrt(i + 1)
        
        return grad


# Hàm tiện ích để tạo instance nhanh
def create_griewank(dim: int = 2, bounds: Optional[List[Tuple[float, float]]] = None) -> Griewank:
    """
    Tạo một instance của bài toán Griewank.

    Args:
        dim: Số chiều
        bounds: Giới hạn tìm kiếm

    Returns:
        Instance của Griewank
    """
    return Griewank(dim=dim, bounds=bounds)
