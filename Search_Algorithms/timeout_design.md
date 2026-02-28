# Thiết kế: Giới hạn thờii gian cho thuật toán

## Mục tiêu

Thêm tính năng timeout cho tất cả các thuật toán để tránh bị treo vô hạn khi chạy experiments.

## Thiết kế tổng quan

### 1. Cập nhật Base Classes

#### SearchAlgorithm (algorithms/base/search_base.py)
- Thêm tham số `timeout` (giây) trong `__init__`
- Thêm phương thức `_check_timeout()` để kiểm tra
- Lưu `start_time` khi bắt đầu search
- Trả về kết quả partial nếu timeout

#### Optimizer (algorithms/base/optimizer_base.py)
- Thêm tham số `timeout` (giây) trong `__init__`
- Thêm phương thức `_check_timeout()` 
- Kiểm tra timeout trong vòng lặp chính

### 2. Cập nhật các thuật toán Classical

#### BFS, DFS, UCS, AStar, Greedy
- Thêm kiểm tra timeout trong vòng lặp while chính
- Trả về kết quả hiện tại nếu timeout

### 3. Cập nhật các thuật toán Local Search

#### HillClimbing, SimulatedAnnealing
- Thêm kiểm tra timeout trong vòng lặp chính
- Trả về best solution hiện tại nếu timeout

### 4. Cập nhật Config Files

Thêm `timeout` vào mỗi algorithm config:
```yaml
algorithms:
  BFS:
    timeout: 30  # giây
  DFS:
    depth_limit: 1000
    timeout: 30
```

## Chi tiết triển khai

### Base Class Updates

```python
# search_base.py
import time

class SearchAlgorithm(ABC):
    def __init__(self, problem, config=None):
        self.problem = problem
        self.config = config or {}
        self.timeout = self.config.get("timeout", 300)  # mặc định 5 phút
        self.start_time = None
        # ...
    
    def _check_timeout(self):
        if self.start_time and (time.time() - self.start_time) > self.timeout:
            return True
        return False
    
    def search(self):
        self.start_time = time.time()
        # ... trong vòng lặp:
        # if self._check_timeout(): return self._build_result(timeout=True)
```

### Algorithm Implementation Updates

```python
# Ví dụ: BFS
def search(self):
    self.start_time = time.time()
    # ...
    while queue:
        if self._check_timeout():
            return self._build_result(timeout=True)
        # ...
```

### Result Handling

Kết quả cần đánh dấu rõ nếu bị timeout:
```python
{
    "solution": ...,  # hoặc None nếu chưa tìm được
    "cost": ...,
    "nodes_expanded": ...,
    "found": False,
    "timeout": True,  # Đánh dấu bị timeout
    "runtime": timeout_value
}
```

## Memory Optimization - Path Reconstruction

### Vấn đề
Các thuật toán search (BFS, DFS, UCS, A*, Greedy) lưu toàn bộ `path` list trong mỗi node của queue/stack/heap, dẫn đến memory usage cao khi không gian trạng thái lớn (đặc biệt với N-Queens).

### Giải pháp: Path Reconstruction
Thay vì lưu path list trong mỗi node, chỉ lưu **parent pointer** và reconstruct path khi tìm thấy goal.

#### Kiến trúc
```python
# Base class thêm:
- parent_map: Dict[state, Tuple[parent_state, step_cost]]
- _record_parent(child, parent, cost): Ghi nhận parent
- _reconstruct_path(start, goal): Backtrack để reconstruct path
- _clear_parent_map(): Giải phóng memory
```

#### Algorithm Refactoring
```python
# Trước: Lưu path list trong queue
queue = deque([(start, [start], 0)])  # (state, path_list, cost)
queue.append((next_state, path + [next_state], cost + step_cost))

# Sau: Chỉ lưu state và cost, dùng parent_map
queue = deque([(start, 0)])  # (state, cost)
parent_map[next_state] = (state, step_cost)
queue.append((next_state, cost + step_cost))

# Khi tìm thấy goal:
solution, cost = self._reconstruct_path(start, goal_state)
```

#### Memory Comparison
| Approach | Memory per node | Queue 100K nodes |
|----------|----------------|------------------|
| Store path list | ~500 bytes | ~50 MB |
| Path Reconstruction | ~50 bytes | ~5 MB |
| **Tiết kiệm** | **~90%** | **~45 MB** |

#### Các thuật toán đã refactor
- [x] BFS
- [x] DFS  
- [x] UCS
- [x] AStar
- [x] Greedy

## Giá trị timeout đề xuất

| Loại bài toán | Timeout đề xuất |
|--------------|-----------------|
| Grid Pathfinding (nhỏ) | 30 giây |
| N-Queens n=8 | 60-120 giây |
| Continuous problems | 60 giây |

## Luồng dữ liệu

```
Config (timeout: 30s)
    ↓
ExperimentRunner
    ↓
Algorithm.__init__ (lưu timeout)
    ↓
Algorithm.search/run
    ↓
Kiểm tra timeout mỗi iteration
    ↓
Trả về kết quả (có thể timeout=True)
    ↓
Logger ghi nhận kết quả + trạng thái timeout
```

## Lưu ý

1. Timeout là giới hạn mềm - thuật toán chỉ kiểm tra ở mỗi iteration
2. Nếu 1 iteration chạy rất lâu, timeout có thể bị vượt quá
3. Cần đảm bảo mọi kết quả đều có flag `timeout` để dễ phân biệt
