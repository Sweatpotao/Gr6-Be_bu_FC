from abc import ABC, abstractmethod
import time

class Optimizer(ABC):
    def __init__(self, problem, config=None):
        self.problem = problem
        self.config = config or {}

        self.max_evals = self.config.get("max_evaluations", float("inf"))
        self.evaluations = 0

        self.best_solution = None
        self.best_fitness = float("inf")

        self.history = []
        self.runtime = 0

    def evaluate(self, x):
        if self.evaluations >= self.max_evals:
            return None
        self.evaluations += 1
        return self.problem.evaluate(x)

    @abstractmethod
    def run(self):
        pass

    def _build_result(self):
        return {
            "final_score": self.best_fitness,
            "best_solution": self.best_solution,
            "history": self.history,
            "evaluations": self.evaluations,
            "runtime": self.runtime,
            "found": True
        }   

