from abc import ABC, abstractmethod

class DiscreteProblem(ABC):

    @abstractmethod
    def get_start_state(self):
        pass

    @abstractmethod
    def is_goal(self, state):
        pass

    @abstractmethod
    def get_neighbors(self, state):
        """
        return: list of (next_state, step_cost)
        """
        pass

    def heuristic(self, state):
        return 0