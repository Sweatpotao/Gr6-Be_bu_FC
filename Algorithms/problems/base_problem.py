<<<<<<< HEAD
from abc import ABC

class BaseProblem(ABC):
    """Marker interface cho mọi bài toán"""
    pass
=======
from abc import ABC, abstractmethod

class BaseProblem(ABC):
    @abstractmethod
    def clone(self):
        pass
>>>>>>> b48bc3d2c962571356227dca0eba443e675d34b7
