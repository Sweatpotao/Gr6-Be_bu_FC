"""
Các hàm tối ưu liên tục (Continuous Optimization Benchmark Functions).

Module này chứa các hàm benchmark phổ biến cho tối ưu liên tục,
bao gồm cả hàm đơn giản và phức tạp để test các thuật toán tối ưu.
"""

from .base_continuous import ContinuousProblem
from .sphere import Sphere, create_sphere
from .rastrigin import Rastrigin, create_rastrigin
from .rosenbrock import Rosenbrock, create_rosenbrock
from .ackley import Ackley, create_ackley
from .griewank import Griewank, create_griewank

__all__ = [
    # Lớp cơ sở
    "ContinuousProblem",
    
    # Các hàm benchmark
    "Sphere",
    "Rastrigin",
    "Rosenbrock",
    "Ackley",
    "Griewank",
    
    # Hàm tiện ích
    "create_sphere",
    "create_rastrigin",
    "create_rosenbrock",
    "create_ackley",
    "create_griewank",
]
