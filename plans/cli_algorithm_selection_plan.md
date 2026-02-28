# Kế hoạch: Hỗ trợ chọn Algorithm từ Command Line

## Mục tiêu
1. Cho phép ngườ dùng chọn thuật toán cụ thể từ command line để test
2. Thêm TSP và Knapsack vào registry để có thể sử dụng qua config

## Phân tích hiện trạng

### Registry hiện tại (experiment/registry.py)
- **Problems**: sphere, ackley, rastrigin, rosenbrock, griewank, grid_pathfinding, n_queens
- **Algorithms**: HillClimbing, SimulatedAnnealing, BFS, DFS, UCS, Greedy, AStar
- **Thiếu**: TSP, Knapsack

### Vấn đề cần giải quyết
- TSP và Knapsack chưa có method `clone()` - cần thiết cho ExperimentRunner
- Chưa có CLI interface để chọn algorithm nhanh

---

## Chi tiết kế hoạch

### 1. Cập nhật Registry - Thêm TSP và Knapsack

**File**: `experiment/registry.py`

**Thay đổi**:
```python
# Thêm imports
from problems.discrete.tsp import TSP
from problems.discrete.knapsack import Knapsack

# Thêm vào PROBLEM_REGISTRY
PROBLEM_REGISTRY = {
    # ... existing entries ...
    "tsp": TSP,
    "knapsack": Knapsack,
}
```

### 2. Thêm method clone() cho TSP và Knapsack

**File**: `problems/discrete/tsp.py`

**Thêm**:
```python
def clone(self):
    return TSP(self.matrix, self.start_city)
```

**File**: `problems/discrete/knapsack.py`

**Thêm**:
```python
def clone(self):
    # Extract values and weights from items
    values = [item[2] for item in self.items]  # item = (ratio, weight, value)
    weights = [item[1] for item in self.items]
    return Knapsack(values, weights, self.capacity)
```

### 3. Tạo file test_single.py - CLI Interface

**File mới**: `test_single.py`

**Chức năng**:
- Nhận tham số problem từ command line
- Nhận danh sách algorithms từ command line (optional - mặc định chạy tất cả)
- Nhận số lần chạy (runs)
- Tạo config động và chạy test

**Interface thiết kế**:
```bash
# Chạy N-Queens với tất cả algorithms
python test_single.py --problem n_queens

# Chạy N-Queens với algorithms cụ thể
python test_single.py --problem n_queens --algorithms BFS,AStar

# Chạy TSP với 10 runs
python test_single.py --problem tsp --runs 10

# Chạy Knapsack với DFS và timeout tùy chỉnh
python test_single.py --problem knapsack --algorithms DFS --timeout 30
```

**Cấu trúc file test_single.py**:
```python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiment.run_experiment import run_experiment
from experiment.logger import save_summary_txt
from experiment.registry import PROBLEM_REGISTRY, ALGORITHM_REGISTRY

def main():
    parser = argparse.ArgumentParser(description='Test single problem with selected algorithms')
    parser.add_argument('--problem', required=True, help='Problem name (n_queens, tsp, knapsack, ...)')
    parser.add_argument('--algorithms', default=None, help='Comma-separated algorithm names (default: all)')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs (default: 5)')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout per run in seconds (default: 60)')
    parser.add_argument('--output', default=None, help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Validate problem
    if args.problem not in PROBLEM_REGISTRY:
        print(f"Error: Unknown problem '{args.problem}'")
        print(f"Available: {list(PROBLEM_REGISTRY.keys())}")
        sys.exit(1)
    
    # Parse algorithms
    if args.algorithms:
        selected_algos = [a.strip() for a in args.algorithms.split(',')]
        # Validate each algorithm
        for algo in selected_algos:
            if algo not in ALGORITHM_REGISTRY:
                print(f"Error: Unknown algorithm '{algo}'")
                print(f"Available: {list(ALGORITHM_REGISTRY.keys())}")
                sys.exit(1)
    else:
        selected_algos = list(ALGORITHM_REGISTRY.keys())
    
    # Build config dict
    config = build_config(args.problem, selected_algos, args.runs, args.timeout)
    
    # Run experiment
    print(f"\n{'='*60}")
    print(f"Testing: {args.problem}")
    print(f"Algorithms: {', '.join(selected_algos)}")
    print(f"Runs: {args.runs}")
    print(f"{'='*60}\n")
    
    results = run_experiment_from_dict(config)
    
    # Print and save results
    print_results(results)
    
    if args.output:
        save_results(results, args.output)

def build_config(problem, algorithms, runs, timeout):
    """Build config dict from CLI arguments"""
    # ... implementation
    pass

def run_experiment_from_dict(config):
    """Run experiment using config dict (no YAML file needed)"""
    # ... implementation
    pass

if __name__ == "__main__":
    main()
```

### 4. Cập nhật run_experiment.py - Hỗ trợ config dict

**File**: `experiment/run_experiment.py`

**Thay đổi**: Thêm hàm `run_experiment_from_dict()` để có thể chạy mà không cần file YAML

```python
def run_experiment_from_dict(config):
    """Run experiment using config dict directly (no YAML file needed).
    
    Args:
        config: Dict with structure:
            {
                "problem": {"name": str, "params": dict},
                "algorithms": {"AlgoName": {"timeout": int, ...}, ...},
                "experiment": {"runs": int}
            }
    """
    # Similar to run_experiment() but uses dict instead of YAML file
    # ... implementation
```

### 5. Cấu hình mặc định cho từng problem

**Tạo hàm** `get_default_problem_params(problem_name)` trong `test_single.py`:

```python
def get_default_problem_params(problem_name):
    """Get default parameters for each problem type."""
    defaults = {
        "n_queens": {"n": 8},
        "tsp": {
            "distance_matrix": [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]],
            "start_city": 0
        },
        "knapsack": {
            "values": [60, 100, 120],
            "weights": [10, 20, 30],
            "capacity": 50
        },
        "grid_pathfinding": {
            "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "start": [0, 0],
            "goal": [2, 2]
        },
        # ... continuous problems
    }
    return defaults.get(problem_name, {})
```

---

## Các bước thực hiện

| # | Task | Files cần sửa | Mức độ |
|---|------|---------------|--------|
| 1 | Thêm TSP/Knapsack imports vào registry | `experiment/registry.py` | Dễ |
| 2 | Thêm method clone() cho TSP | `problems/discrete/tsp.py` | Dễ |
| 3 | Thêm method clone() cho Knapsack | `problems/discrete/knapsack.py` | Dễ |
| 4 | Thêm hàm run_experiment_from_dict() | `experiment/run_experiment.py` | Trung bình |
| 5 | Tạo test_single.py với CLI | `test_single.py` (mới) | Trung bình |

---

## Ví dụ sử dụng sau khi hoàn thành

```bash
# 1. Test N-Queens với A* và BFS
python test_single.py --problem n_queens --algorithms AStar,BFS

# 2. Test TSP với tất cả algorithms, 10 runs
python test_single.py --problem tsp --runs 10

# 3. Test Knapsack với Greedy và UCS, timeout 30s
python test_single.py --problem knapsack --algorithms Greedy,UCS --timeout 30

# 4. Test và lưu kết quả ra file
python test_single.py --problem n_queens --algorithms AStar --output results.txt
```

---

## Lưu ý kỹ thuật

1. **Problem Registry**: TSP và Knapsack cần được import và đăng ký trước khi sử dụng
2. **Clone method**: Cần đảm bảo clone() tạo ra instance mới hoàn toàn độc lập
3. **Error handling**: Cần validate problem và algorithm names trước khi chạy
4. **Backward compatibility**: Các thay đổi không ảnh hưởng đến code hiện tại
