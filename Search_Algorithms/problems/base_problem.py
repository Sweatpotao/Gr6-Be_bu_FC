from abc import ABC, abstractmethod

class BaseProblem(ABC):
    @abstractmethod
    def clone(self):
        pass
