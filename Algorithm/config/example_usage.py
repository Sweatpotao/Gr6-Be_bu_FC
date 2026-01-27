"""
Ví dụ sử dụng hệ thống cấu hình thuật toán.
Minh họa cách tải và sử dụng cấu hình từ file classical.yaml.
"""

import sys
import io

# Thiết lập encoding UTF-8 cho stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from Algorithm.config import get_config, load_algorithm
from Algorithm.Problems.discrete.grid_pathfinding import GridPathFinding


def example_usage():
    """Ví dụ sử dụng hệ thống cấu hình"""
    print("=== Ví dụ Hệ thống Cấu hình Thuật toán ===\n")

    # Lấy instance cấu hình
    config = get_config()

    # 1. Lấy tất cả các thuật toán có sẵn
    print("1. Các thuật toán có sẵn:")
    for algo_name, algo_config in config.algorithms.items():
        print(f"   - {algo_name}: {algo_config['name']}")
    print()

    # 2. Lấy cấu hình toàn cục
    print("2. Cấu hình toàn cục:")
    print(f"   Thuật toán mặc định: {config.global_config['default_algorithm']}")
    print(
        f"   Số nút mở rộng tối đa: {config.global_config['termination']['max_nodes_expanded']}")
    print(
        f"   Thờii gian tối đa (giây): {config.global_config['termination']['max_time_seconds']}")
    print()

    # 3. Lấy chi tiết thuật toán
    print("3. Chi tiết thuật toán (A*):")
    astar_config = config.get_algorithm_config("astar")
    if astar_config:
        print(f"   Tên: {astar_config['name']}")
        print(f"   Loại: {astar_config['type']}")
        print(f"   Mô tả: {astar_config['description']}")
        print(f"   Tính đầy đủ: {astar_config['properties']['completeness']}")
        print(f"   Tính tối ưu: {astar_config['properties']['optimality']}")
        print(
            f"   Độ phức tạp thờii gian: {astar_config['properties']['time_complexity']}")
        print(
            f"   Độ phức tạp không gian: {astar_config['properties']['space_complexity']}")

    print()
    print("4. Chi tiết thuật toán (BFS):")
    bfs_config = config.get_algorithm_config("bfs")
    if bfs_config:
        print(f"   Tên: {bfs_config['name']}")
        print(f"   Loại: {bfs_config['type']}")
        print(f"   Mô tả: {bfs_config['description']}")
    print()

    # 5. Lấy cấu hình riêng cho bài toán
    print("5. Cấu hình bài toán tìm đường trên lưới:")
    grid_config = config.get_problem_config("grid_pathfinding")
    if grid_config:
        print(f"   Thuật toán mặc định: {grid_config['default_algorithm']}")
        print(f"   Heuristic: {grid_config['heuristic']}")
        print(f"   Kiểu di chuyển: {grid_config['movement']}")
        print(f"   Chi phí: {grid_config['cost']}")
    print()

    # 6. Tải lớp thuật toán động
    print("6. Tải thuật toán động:")
    try:
        algo_class = load_algorithm("astar")
        print(f"   Tải thành công: {algo_class.__name__}")
    except Exception as e:
        print(f"   Lỗi khi tải thuật toán: {e}")
    print()

    # 7. Ví dụ: Tạo bài toán lưới và sử dụng thuật toán mặc định
    print("7. Ví dụ: Giải bài toán tìm đường trên lưới")

    # Lưới đơn giản: 0 = ô trống, 1 = chướng ngại vật
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    goal = (4, 4)

    problem = GridPathFinding(
        grid, start, goal, heuristic_type="manhattan")

    # Lấy thuật toán mặc định cho bài toán tìm đường
    default_algo_name = config.get_default_algorithm("grid_pathfinding")
    print(
        f"   Sử dụng thuật toán mặc định cho tìm đường: {default_algo_name}")

    algo_class = load_algorithm(default_algo_name)
    algorithm = algo_class(problem)

    result = algorithm.search()

    if result.success:
        print(f"   Thành công! Tìm thấy đường đi với {len(result.solution)} bước")
        print(f"   Chi phí: {result.cost}")
        print(f"   Số nút đã mở rộng: {result.nodes_expanded}")
        print(f"   Thờii gian thực thi: {result.time_elapsed:.4f} giây")
    else:
        print("   Không tìm thấy đường đi")

if __name__ == "__main__":
    example_usage()
