"""
Sphere Function - Hàm cầu (hàm De Jong F1).

Một trong những hàm benchmark đơn giản nhất cho tối ưu liên tục.
Là hàm lồi, có global minimum tại gốc tọa độ.
"""

import numpy as np
from typing import List, Tuple, Optional
from problems.continous.base_continuous import ContinuousProblem


class Sphere(ContinuousProblem):
    """
    Sphere Function: f(x) = sum(x_i^2)
    
    Hàm cầu đơn giản, là hàm lồi với global minimum tại x = 0.
    
    Đặc điểm:
    - Global minimum: f(0, 0, ..., 0) = 0
    - Miền tìm kiếm thường dùng: [-5.12, 5.12]^d
    - Hàm lồi, liên tục, không có local minimum
    - Gradient tại x: grad(x) = 2*x
    
    Công thức:
        f(x) = Σ(x_i²) với i = 1 đến dim
    
    Attributes:
        dim: Số chiều của không gian tìm kiếm
        bounds: Giới hạn của không gian tìm kiếm
    
    Examples:
        >>> sphere = Sphere(dim=2)
        >>> x = np.array([1.0, 2.0])
        >>> sphere.evaluate(x)
        5.0
    """

    def __init__(self, dim: int = 2, bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Khởi tạo bài toán Sphere.

        Args:
            dim: Số chiều của không gian tìm kiếm
            bounds: Giới hạn tìm kiếm, mặc định [-5.12, 5.12]^dim
        """
        if bounds is None:
            bounds = [(-5.12, 5.12)] * dim
        
        super().__init__(dim=dim, bounds=bounds, name="Sphere")

    def evaluate(self, x: np.ndarray) -> float:
        """
        Đánh giá giá trị hàm Sphere tại điểm x.

        Args:
            x: Vector điểm trong không gian tìm kiếm, shape (dim,)

        Returns:
            Giá trị f(x) = sum(x_i^2)

        Raises:
            ValueError: Nếu kích thước của x không khớp với dim
        """
        x = np.asarray(x)
        if x.shape[0] != self.dim:
            raise ValueError(f"Kích thước của x ({x.shape[0]}) không khớp với dim ({self.dim})")
        
        return float(np.sum(x ** 2))

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Trả về điểm tối ưu toàn cục của hàm Sphere.

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
        Tính gradient của hàm Sphere tại điểm x.

        Args:
            x: Vector điểm cần tính gradient

        Returns:
            Vector gradient: grad(x) = 2*x
        """
        x = np.asarray(x)
        return 2.0 * x


# Hàm tiện ích để tạo instance nhanh
def create_sphere(dim: int = 2, bounds: Optional[List[Tuple[float, float]]] = None) -> Sphere:
    """
    Tạo một instance của bài toán Sphere.

    Args:
        dim: Số chiều
        bounds: Giới hạn tìm kiếm

    Returns:
        Instance của Sphere
    """
    return Sphere(dim=dim, bounds=bounds)
