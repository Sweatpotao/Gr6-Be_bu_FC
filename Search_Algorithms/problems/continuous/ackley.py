"""
Ackley Function.

Hàm có một global minimum duy nhất và rất nhiều local minimum,
thường dùng để test khả năng tìm kiếm toàn cục của thuật toán.
"""

import numpy as np
from typing import Tuple
from problems.continuous.continuous_problem import ContinuousProblem


class Ackley(ContinuousProblem):
    """
    Ackley Function.
    
    f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    
    Hàm có một global minimum tại x = 0 và rất nhiều local minimum.
    Bề mặt có dạng như "miệng núi lửa" với nhiều điểm cực trị địa phương.
    
    Đặc điểm:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Miền tìm kiếm thường dùng: [-32.768, 32.768]^d hoặc [-5, 5]^d
    - Có rất nhiều local minimum
    - Bề mặt gần như phẳng xa global minimum
    
    Công thức:
        f(x) = -20*exp(-0.2*sqrt(1/d * Σx_i²)) 
               - exp(1/d * Σcos(2π*x_i)) + 20 + e
        với i = 1 đến dim
    
    Attributes:
        dim: Số chiều của không gian tìm kiếm
        bounds: Giới hạn của không gian tìm kiếm
        a, b, c: Các tham số điều chỉnh (mặc định 20, 0.2, 2π)
    
    Examples:
        >>> ackley = Ackley(dim=2)
        >>> x = np.array([0.0, 0.0])
        >>> ackley.evaluate(x)
        0.0
    """

    def __init__(self, dim: int = 2, bounds: Tuple[float, float] = (-32.768, 32.768),
                 a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi):
        """
        Khởi tạo bài toán Ackley.

        Args:
            dim: Số chiều của không gian tìm kiếm
            bounds: Giới hạn tìm kiếm, mặc định [-32.768, 32.768]
            a: Tham số điều chỉnh độ sâu (mặc định 20)
            b: Tham số điều chỉnh độ rộng (mặc định 0.2)
            c: Tham số tần số (mặc định 2π)
        """
        self.dim = dim
        self.bounds = tuple(bounds)
        self.name = "Ackley"
        self.a = a
        self.b = b
        self.c = c
    
    def get_dimension(self):
        return self.dim
    
    def get_bounds(self):
        return self.bounds
    
    def clone(self):
        return Ackley(self.dim, self.bounds, self.a, self.b, self.c)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Đánh giá giá trị hàm Ackley tại điểm x.

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
        sum_cos = np.sum(np.cos(self.c * x))
        
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / self.dim))
        term2 = -np.exp(sum_cos / self.dim)
        
        return float(term1 + term2 + self.a + np.e)

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Trả về điểm tối ưu toàn cục của hàm Ackley.

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
        Tính gradient của hàm Ackley tại điểm x.

        Args:
            x: Vector điểm cần tính gradient

        Returns:
            Vector gradient
        """
        x = np.asarray(x)
        
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(self.c * x))
        
        sqrt_term = np.sqrt(sum_sq / self.dim)
        exp_term1 = np.exp(-self.b * sqrt_term)
        exp_term2 = np.exp(sum_cos / self.dim)
        
        if sqrt_term > 1e-10:
            grad_sq = (self.a * self.b * exp_term1) / (self.dim * sqrt_term)
        else:
            grad_sq = 0.0
        
        grad_cos = (self.c * exp_term2 / self.dim) * np.sin(self.c * x)
        
        return grad_sq * x + grad_cos


# Hàm tiện ích để tạo instance nhanh
def create_ackley(dim: int = 2, bounds: Tuple[float, float] = (-32.768, 32.768),
                  a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> Ackley:
    """
    Tạo một instance của bài toán Ackley.

    Args:
        dim: Số chiều
        bounds: Giới hạn tìm kiếm
        a: Tham số điều chỉnh độ sâu
        b: Tham số điều chỉnh độ rộng
        c: Tham số tần số

    Returns:
        Instance của Ackley
    """
    return Ackley(dim=dim, bounds=bounds, a=a, b=b, c=c)