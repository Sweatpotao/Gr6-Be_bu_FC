from problems.discrete_problem import DiscreteProblem

class Knapsack(DiscreteProblem):
    def __init__(self, values, weights, capacity):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.n = len(values)

    def get_start_state(self):
        # State: (index, current_weight)
        return (0, 0)

    def is_goal(self, state):
        idx, _ = state
        return idx == self.n

    def get_neighbors(self, state):
        idx, current_weight = state
        if idx >= self.n:
            return []

        neighbors = []
        # Option 1: Không chọn item nào
        neighbors.append(((idx + 1, current_weight), 0))

        # Option 2: Chọn item vừa capacity
        if current_weight + self.weights[idx] <= self.capacity:
            # Vì SearchAlgorithm tìm cost nhỏ nhất, ta dùng số âm của value làm cost
            neighbors.append(((idx + 1, current_weight + self.weights[idx]), -self.values[idx]))

        return neighbors

    def heuristic(self, state):
        idx, current_weight = state
        remaining_capacity = self.capacity - current_weight

        # Tính heuristic greedy theo value/weight
        items = [(self.values[i]/self.weights[i], self.weights[i], self.values[i]) for i in range(idx, self.n)]
        items.sort(reverse=True)  # ưu tiên value/weight cao

        est_value = 0
        for ratio, w, v in items:
            if w <= remaining_capacity:
                remaining_capacity -= w
                est_value += v

        return -est_value  # cost = -value