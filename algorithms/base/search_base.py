from abc import ABC, abstractmethod

class SearchAlgorithm(ABC):
    def __init__(self, problem, config = None):
        self.problem = problem
        self.config = config

        self.solution = None
        self.cost = float("inf")
        self.nodes_expanded = 0

    #Hàm này bắt buộc con kế thừa phải cài đặt lại
    @abstractmethod
    def search(self):
        pass

    def _build_result(self):
        return {
            "solution": self.solution,
            "cost": self.cost,
            "nodes_expanded": self.nodes_expanded
        }