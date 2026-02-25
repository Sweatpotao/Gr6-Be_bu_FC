from abc import ABC, abstractmethod
import numpy as np
from problems.base_problem import BaseProblem

class ContinuousProblem(BaseProblem, ABC):

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_bounds(self):
        """ return (low, high) """
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        pass

    def initial_solution(self):
        low, high = self.get_bounds()
        return np.random.uniform(low, high, self.get_dimension())