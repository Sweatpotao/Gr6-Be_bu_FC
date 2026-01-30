import numpy as np
from problems.continuous.continuous_problem import ContinuousProblem

class SphereFunction(ContinuousProblem):
    def __init__(self, dim=10, bounds=(-5.12, 5.12)):
        self.dim = dim
        self.bounds = bounds

    def get_dimension(self):
        return self.dim

    def get_bounds(self):
        return self.bounds

    def evaluate(self, x):
        return float(np.sum(x ** 2))

    def clone(self):
        return SphereFunction(self.dim, self.bounds)
