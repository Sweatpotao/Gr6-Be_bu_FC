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
        self.timeout = self.config.get("timeout", 300)  # Mặc định 5 phút
        self.start_time = None
        self.timed_out = False

    def _check_timeout(self):
        """Kiểm tra xem đã vượt quá thờii gian timeout chưa."""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            self.timed_out = True
            return True
        return False

    def evaluate(self, x):
        if self.evaluations >= self.max_evals:
            return None
        self.evaluations += 1
        return self.problem.evaluate(x)

    @abstractmethod
    def run(self):
        pass

    def _build_result(self):
        # Tính runtime nếu chưa có
        if self.runtime == 0 and self.start_time is not None:
            self.runtime = time.time() - self.start_time

        return {
            "final_score": self.best_fitness,
            "best_solution": self.best_solution,
            "history": self.history,
            "evaluations": self.evaluations,
            "runtime": self.runtime,
            "found": True,
            "timeout": self.timed_out
        }

