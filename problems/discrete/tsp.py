from problems.discrete_problem import DiscreteProblem

class TSP(DiscreteProblem):
    def __init__(self, distance_matrix, start_city=0):
        self.matrix = distance_matrix
        self.n = len(distance_matrix)
        self.start_city = start_city

    def get_start_state(self):
        # State: thành phố hiện tại, tuple các thành phố đã đi qua
        return (self.start_city, (self.start_city,))

    def is_goal(self, state):
        curr, visited = state
        # Goal: Đã đi qua tất cả thành phố và quay về điểm bắt đầu
        return len(visited) == self.n + 1 and curr == self.start_city

    def get_neighbors(self, state):
        curr, visited_mask = state
        neighbors = []
        
        # Dùng bitmask 
        # Nếu đã đi đủ n thành phố, quay về điểm bắt đầu
        if bin(visited_mask).count("1") == self.n:
            cost = self.matrix[curr][self.start_city]
            # TH: ko có 'đường đi' giữa 2 thành phố -> cost = vô cùng
            if cost < float('inf'):
                neighbors.append(((self.start_city, visited_mask | (1 << self.start_city)), cost))
            return neighbors

        # Thử đi đến các thành phố chưa thăm
        for next_city in range(self.n):
            if not (visited_mask & (1 << next_city)):
                cost = self.matrix[curr][next_city]
                new_mask = visited_mask | (1 << next_city)
                neighbors.append(((next_city, new_mask), cost))

        return neighbors

    def heuristic(self, state):
        # Heuristic: Khoảng cách về nhà
        # Nếu không (inf), trả về 0 để thuật toán vẫn tiếp tục tìm các đường vòng khác.
        return self.matrix[state[0]][self.start_city] if self.matrix[state[0]][self.start_city] != float('inf') else 0