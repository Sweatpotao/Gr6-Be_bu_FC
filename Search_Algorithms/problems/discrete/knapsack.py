from problems.discrete.discrete_problem import DiscreteProblem

class Knapsack(DiscreteProblem):
    def __init__(self, values, weights, capacity):
        self.capacity = capacity
        self.n = len(values)

         # Lưu item dạng tuple: (ratio, weight, value)
        self.items = []
        for v, w in zip(values, weights):
            if w == 0:
                ratio = float("inf")  # tránh chia 0
            else:
                ratio = v / w
            self.items.append((ratio, w, v))

        # Sort giảm dần theo ratio để heuristic fractional nhanh
        self.items.sort(reverse=True)


    def get_start_state(self):
        # State: (index - sorted items, current_weight)
        return (0, 0)

    def is_goal(self, state):
        idx, _ = state
        return idx == self.n

    def get_neighbors(self, state):
        idx, current_weight = state
        if idx >= self.n:
            return []

        neighbors = []
        ratio, w, v = self.items[idx]

        # Option 1: Không chọn item nào
        neighbors.append(((idx + 1, current_weight), 0))

        # Option 2: Chọn item vừa capacity
        if current_weight + w <= self.capacity:
            neighbors.append(((idx + 1, current_weight + w), -v))

        return neighbors

    def heuristic(self, state):
        # Heuristic = - (upper bound value còn có thể lấy thêm)
        # Fractional knapsack => bound lạc quan, giúp A* chạy nhanh hơn.
        
        idx, current_weight = state
        remaining_cap = self.capacity - current_weight
        est_additional_value = 0.0

        for i in range(idx, self.n):
            ratio, w, v = self.items[i]

            if remaining_cap <= 0:
                break

            if w <= remaining_cap:
                remaining_cap -= w
                est_additional_value += v
            else:
                # lấy 1 phần item cuối (fractional)
                if w != 0:
                    est_additional_value += ratio * remaining_cap
                break

        return -est_additional_value