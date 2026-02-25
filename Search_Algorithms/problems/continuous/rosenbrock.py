"""
Rosenbrock Function - Hàm thung lũng Rosenbrock (hàm chuối).

Hàm có dạng thung lũng hẹp, rất khó tối ưu vì gradient nhỏ
ở khu vực gần optimum.
"""

import numpy as np
from typing import Tuple
from problems.continuous.continuous_problem import ContinuousProblem


class Rosenbrock(ContinuousProblem):
    """
    Rosenbrock Function (Banana Function).
    
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Hàm có dạng thung lũng cong hẹp, rất khó tối ưu.
    Gradient rất nhỏ ở khu vực gần optimum.
    
    Đặc điểm:
    - Global minimum: f(1, 1, ..., 1) = 0
    - Miền tìm kiếm thường dùng: [-5, 10]^d hoặc [-2.048, 2.048]^d
    - Còn gọi là "banana function" do đường đồng mức có dạng chuối
    - Rất khó tối ưu do gradient nhỏ gần optimum
    
    Công thức:
        f(x) = Σ[100*(x_{i+1} - x_i²)² + (1 - x_i)²]
        với i = 1 đến dim-1
    
    Attributes:
        dim: Số chiều của không gian tìm kiếm (phải >= 2)
        bounds: Tuple (low, high) áp dụng chung cho mọi chiều
    
    Examples:
        >>> rosenbrock = Rosenbrock(dim=2)
        >>> x = np.array([1.0, 1.0])
        >>> rosenbrock.evaluate(x)
        0.0
    """

    def __init__(self, dim: int = 2, bounds: Tuple[float, float] = (-5.0, 10.0)):
        """
        Khởi tạo bài toán Rosenbrock.

        Args:
            dim: Số chiều của không gian tìm kiếm (phải >= 2)
            bounds: Giới hạn tìm kiếm dạng (low, high)
                    Áp dụng cho tất cả các chiều.
        """
        # Kiểm tra điều kiện tối thiểu
        if dim < 2:
            raise ValueError("Rosenbrock function yêu cầu dim >= 2")
        
        self.dim = dim

        # Lưu bounds dưới dạng tuple (low, high)
        # Đồng nhất với toàn bộ framework continuous problems
        self.bounds = tuple(bounds)

        self.name = "Rosenbrock"
    
    def get_dimension(self):
        """
        Trả về số chiều của bài toán.
        """
        return self.dim
    
    def get_bounds(self):
        """
        Trả về tuple (low, high) của không gian tìm kiếm.
        Áp dụng chung cho mọi chiều.
        """
        return self.bounds
    
    def clone(self):
        """
        Tạo bản sao của bài toán với cùng dim và bounds.
        """
        return Rosenbrock(self.dim, self.bounds)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Đánh giá giá trị hàm Rosenbrock tại điểm x.

        Args:
            x: Vector điểm trong không gian tìm kiếm, shape (dim,)

        Returns:
            Giá trị f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

        Raises:
            ValueError: Nếu kích thước của x không khớp với dim
        """
        x = np.asarray(x)

        # Kiểm tra kích thước đầu vào
        if x.shape[0] != self.dim:
            raise ValueError(
                f"Kích thước của x ({x.shape[0]}) không khớp với dim ({self.dim})"
            )
        
        # Tính tổng cho i = 0 đến dim-2
        result = 0.0
        for i in range(self.dim - 1):
            result += (
                100.0 * (x[i+1] - x[i]**2)**2
                + (1.0 - x[i])**2
            )
        
        return float(result)

    def get_optimum(self) -> Tuple[np.ndarray, float]:
        """
        Trả về điểm tối ưu toàn cục của hàm Rosenbrock.

        Returns:
            Tuple (x_opt, f_opt) trong đó:
            - x_opt: Vector [1, 1, ..., 1]
            - f_opt: 0.0
        """
        x_opt = np.ones(self.dim)
        f_opt = 0.0
        return x_opt, f_opt

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Tính gradient của hàm Rosenbrock tại điểm x.

        Args:
            x: Vector điểm cần tính gradient

        Returns:
            Vector gradient cùng kích thước với x
        """
        x = np.asarray(x)

        # Khởi tạo vector gradient
        grad = np.zeros(self.dim)
        
        # Gradient cho chiều đầu tiên (i = 0)
        grad[0] = (
            -400.0 * x[0] * (x[1] - x[0]**2)
            - 2.0 * (1.0 - x[0])
        )
        
        # Gradient cho các chiều giữa
        for i in range(1, self.dim - 1):
            grad[i] = (
                200.0 * (x[i] - x[i-1]**2)
                - 400.0 * x[i] * (x[i+1] - x[i]**2)
                - 2.0 * (1.0 - x[i])
            )
        
        # Gradient cho chiều cuối cùng
        grad[-1] = 200.0 * (x[-1] - x[-2]**2)
        
        return grad


# Hàm tiện ích để tạo instance nhanh
def create_rosenbrock(
    dim: int = 2,
    bounds: Tuple[float, float] = (-5.0, 10.0)
) -> Rosenbrock:
    """
    Tạo một instance của bài toán Rosenbrock.

    Args:
        dim: Số chiều (phải >= 2)
        bounds: Giới hạn tìm kiếm dạng (low, high)

    Returns:
        Instance của Rosenbrock
    """
    return Rosenbrock(dim=dim, bounds=bounds)