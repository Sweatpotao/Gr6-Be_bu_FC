"""
Module cấu hình cho các thuật toán tìm kiếm.
Cung cấp các hàm để tải và quản lý cấu hình từ file YAML.
"""

import yaml
import os
import importlib
from typing import Dict, Any, Optional


class AlgorithmConfig:
    """Quản lý cấu hình thuật toán từ file YAML."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Tải cấu hình từ file classical.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), 'classical.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    @property
    def algorithms(self) -> Dict[str, Any]:
        """Trả về dictionary các thuật toán."""
        return self._config.get('algorithms', {})

    @property
    def global_config(self) -> Dict[str, Any]:
        """Trả về cấu hình toàn cục."""
        return self._config.get('global', {})

    def get_algorithm_config(self, algorithm_name: str) -> Optional[Dict[str, Any]]:
        """Lấy cấu hình của một thuật toán cụ thể."""
        return self.algorithms.get(algorithm_name)

    def get_problem_config(self, problem_name: str) -> Optional[Dict[str, Any]]:
        """Lấy cấu hình cho một bài toán cụ thể."""
        # Tạm thờii trả về cấu hình mặc định cho grid_pathfinding
        if problem_name == "grid_pathfinding":
            return {
                "default_algorithm": "astar",
                "heuristic": "manhattan",
                "movement": "4-directional",
                "cost": "uniform"
            }
        return None

    def get_default_algorithm(self, problem_name: str) -> str:
        """Lấy thuật toán mặc định cho một bài toán."""
        problem_config = self.get_problem_config(problem_name)
        if problem_config and "default_algorithm" in problem_config:
            return problem_config["default_algorithm"]
        return self.global_config.get("default_algorithm", "astar")


def get_config() -> AlgorithmConfig:
    """Trả về instance cấu hình (Singleton pattern)."""
    return AlgorithmConfig()


def load_algorithm(algorithm_name: str):
    """
    Tải động lớp thuật toán dựa trên tên.

    Args:
        algorithm_name: Tên thuật toán (bfs, dfs, astar, v.v.)

    Returns:
        Lớp thuật toán đã được import

    Raises:
        ValueError: Nếu thuật toán không tồn tại
        ImportError: Nếu không thể import module
    """
    config = get_config()
    algo_config = config.get_algorithm_config(algorithm_name)

    if not algo_config:
        raise ValueError(f"Thuật toán '{algorithm_name}' không tồn tại trong cấu hình")

    module_path = algo_config.get('module')
    class_name = algo_config.get('class')

    if not module_path or not class_name:
        raise ValueError(f"Cấu hình thuật toán '{algorithm_name}' thiếu module hoặc class")

    try:
        module = importlib.import_module(module_path)
        algo_class = getattr(module, class_name)
        return algo_class
    except ImportError as e:
        raise ImportError(f"Không thể import module '{module_path}': {e}")
    except AttributeError as e:
        raise ImportError(f"Không tìm thấy class '{class_name}' trong module '{module_path}': {e}")
