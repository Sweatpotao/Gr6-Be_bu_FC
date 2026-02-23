from problems.discrete.discrete_problem import DiscreteProblem

class TSP(DiscreteProblem):
    def __init__(self, distance_matrix, start_city=0):
        self.matrix = distance_matrix
        self.n = len(distance_matrix)
        self.start_city = start_city
        self.all_visited = (1 << self.n) - 1
        self.done_bit = (1 << self.n)

    def get_start_state(self):
        # State: (current_city, visited_bitmask)
        return (self.start_city, 1 << self.start_city)

    def is_goal(self, state):
        curr, visited_mask = state
        # Goal: Đã đi qua tất cả thành phố và quay về start_city
        return visited_mask == (self.done_bit | self.all_visited) and curr == self.start_city

    def get_neighbors(self, state):
        curr, visited_mask = state
        neighbors = []

        # Nếu đã quay về rồi -> kết thúc, không sinh hàng xóm nữa
        if visited_mask & self.done_bit:
            return []
        
        # Nếu đã thăm tất cả thành phố, quay về start
        if (visited_mask & self.all_visited) == self.all_visited:
            cost = self.matrix[curr][self.start_city]
            if cost < float('inf'):
                # Bật bit thứ n, tránh loop
                new_mask = visited_mask | (1 << self.n)
                neighbors.append(((self.start_city, new_mask), cost))
            return neighbors

        # Đi đến các thành phố chưa thăm
        for next_city in range(self.n):
            if not (visited_mask & (1 << next_city)):
                cost = self.matrix[curr][next_city]
                if cost < float('inf'):
                    new_mask = visited_mask | (1 << next_city)
                    neighbors.append(((next_city, new_mask), cost))

        return neighbors

    def heuristic(self, state):
        # h(n) = min(curr->next_city) + min(curr->home_city)
        curr, visited_mask = state

        # Nếu đã done -> h = 0
        if visited_mask & self.done_bit:
            return 0

        unvisited = [i for i in range(self.n) if not (visited_mask & (1 << i))]

        # Nếu không -> về start (nhưng thực tế case này sẽ đi qua nhánh quay về ở get_neighbors)
        if not unvisited:
            dist = self.matrix[curr][self.start_city]
            return dist if dist < float('inf') else 0

        min_from_curr = min(self.matrix[curr][i] for i in unvisited)
        min_to_start = min(self.matrix[i][self.start_city] for i in unvisited)

        if min_from_curr == float('inf'):
            min_from_curr = 0
        if min_to_start == float('inf'):
            min_to_start = 0

        return min_from_curr + min_to_start

    def clone(self):
        """Create a fresh copy of this TSP problem for independent runs."""
        return TSP(self.matrix, self.start_city)