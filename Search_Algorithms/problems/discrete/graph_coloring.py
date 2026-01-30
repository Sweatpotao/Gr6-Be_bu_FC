from problems.discrete.discrete_problem import DiscreteProblem

class GraphColoring(DiscreteProblem):
    def __init__(self, adjacency_matrix, num_colors):
        self.adj = adjacency_matrix
        self.n = len(adjacency_matrix)
        self.num_colors = num_colors

    def get_start_state(self):
        # State: tuple màu của từng node (-1 là chưa tô)
        return tuple([-1] * self.n)

    def is_goal(self, state):
        # Nếu đã tô hết các node (vì get_neighbors đã check valid nên ko cần check lại conflict)
        return -1 not in state

    def get_neighbors(self, state):
        # Tìm node đầu tiên chưa được tô
        node_idx = -1
        for i in range(self.n):
            if state[i] == -1:
                node_idx = i
                break
        
        if node_idx == -1: return []

        neighbors = []
        for color in range(self.num_colors):
            # Kiểm tra xem có node lân cận nào trùng màu không
            if self._is_valid(state, node_idx, color):
                new_state = list(state)
                new_state[node_idx] = color
                neighbors.append((tuple(new_state), 1)) # step_cost là 1 đơn vị gán màu
        return neighbors

    def _is_valid(self, state, node, color):
        for neighbor in range(self.n):
            if self.adj[node][neighbor] == 1 and state[neighbor] == color:
                return False
        return True
    
    def heuristic(self, state):
        return state.count(-1)