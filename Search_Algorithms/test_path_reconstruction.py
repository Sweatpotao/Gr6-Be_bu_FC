"""
Test script for Path Reconstruction optimization.
Verifies that all algorithms still work correctly after refactoring.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.classical.bfs import BFS
from algorithms.classical.dfs import DFS
from algorithms.classical.ucs import UCS
from algorithms.classical.greedy import Greedy
from algorithms.classical.astar import AStar
from problems.discrete.n_queens import NQueens
from problems.discrete.grid_pathfinding import GridPathfinding

def test_nqueens():
    """Test N-Queens with n=5 (small enough to solve quickly)"""
    print("\n" + "="*60)
    print("Testing N-Queens (n=5)")
    print("="*60)
    
    problem = NQueens(n=5)
    algorithms = {
        'BFS': BFS,
        'DFS': DFS,
        'UCS': UCS,
        'Greedy': Greedy,
        'AStar': AStar
    }
    
    for name, AlgoClass in algorithms.items():
        print(f"\n{name}:")
        try:
            prob = NQueens(n=5)
            algo = AlgoClass(prob, {"timeout": 30, "depth_limit": 1000})
            result = algo.search()
            
            if result["found"]:
                print(f"  [OK] Found solution!")
                print(f"    Nodes expanded: {result['nodes_expanded']}")
                print(f"    Runtime: {result['runtime']:.4f}s")
                print(f"    Solution length: {len(result['solution']) if result['solution'] else 0}")
                # Verify solution is valid
                if result['solution']:
                    final_state = result['solution'][-1]
                    is_valid = prob.is_goal(final_state)
                    print(f"    Solution valid: {is_valid}")
            else:
                print(f"  [NO] No solution found")
                print(f"    Timeout: {result.get('timeout', False)}")
                print(f"    Nodes expanded: {result['nodes_expanded']}")
        except Exception as e:
            print(f"  [ERR] Error: {e}")
            import traceback
            traceback.print_exc()

def test_grid_pathfinding():
    """Test Grid Pathfinding"""
    print("\n" + "="*60)
    print("Testing Grid Pathfinding")
    print("="*60)
    
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ]
    
    algorithms = {
        'BFS': BFS,
        'DFS': DFS,
        'UCS': UCS,
        'Greedy': Greedy,
        'AStar': AStar
    }
    
    for name, AlgoClass in algorithms.items():
        print(f"\n{name}:")
        try:
            prob = GridPathfinding(grid, start=(0, 0), goal=(3, 3))
            algo = AlgoClass(prob, {"timeout": 30, "depth_limit": 100})
            result = algo.search()
            
            if result["found"]:
                print(f"  [OK] Found path!")
                print(f"    Path length: {len(result['solution'])}")
                print(f"    Cost: {result.get('final_score', 'N/A')}")
                print(f"    Nodes expanded: {result['nodes_expanded']}")
                print(f"    Runtime: {result['runtime']:.4f}s")
            else:
                print(f"  [NO] No path found")
                print(f"    Timeout: {result.get('timeout', False)}")
                print(f"    Nodes expanded: {result['nodes_expanded']}")
        except Exception as e:
            print(f"  [ERR] Error: {e}")
            import traceback
            traceback.print_exc()

def test_memory_efficiency():
    """Test memory efficiency with larger N-Queens instance"""
    print("\n" + "="*60)
    print("Testing Memory Efficiency (N-Queens n=8 with timeout)")
    print("="*60)
    
    # Use short timeout to test memory doesn't explode
    import time
    
    algorithms = {
        'BFS': BFS,
        'DFS': DFS,
        'UCS': UCS,
        'Greedy': Greedy,
        'AStar': AStar
    }
    
    print("\nRunning with 2-second timeout (should timeout for most algorithms)...")
    
    for name, AlgoClass in algorithms.items():
        print(f"\n{name}:")
        try:
            import os
            import psutil
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            prob = NQueens(n=8)
            algo = AlgoClass(prob, {"timeout": 2, "depth_limit": 100})
            
            start_time = time.time()
            result = algo.search()
            elapsed = time.time() - start_time
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            print(f"  Runtime: {elapsed:.2f}s")
            print(f"  Timeout: {result.get('timeout', False)}")
            print(f"  Nodes expanded: {result['nodes_expanded']}")
            print(f"  Memory used: {mem_used:.2f} MB")
            
            if result.get('timeout'):
                print(f"  [OK] Handled timeout gracefully")
            elif result['found']:
                print(f"  [OK] Found solution!")
            else:
                print(f"  [NO] No solution")
                
        except Exception as e:
            print(f"  [ERR] Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("PATH RECONSTRUCTION TEST SUITE")
    print("="*60)
    
    test_nqueens()
    test_grid_pathfinding()
    
    # Only run memory test if psutil is available
    try:
        import psutil
        test_memory_efficiency()
    except ImportError:
        print("\n\nNote: psutil not installed, skipping memory efficiency test")
        print("Install with: pip install psutil")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)