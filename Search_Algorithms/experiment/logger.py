import os
from datetime import datetime

def save_summary_txt(algo_name, summary, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Experiment time : {datetime.now()}\n")
        f.write(f"Algorithm       : {algo_name}\n")
        f.write(f"Runs            : {summary['runs']}\n")

        # Kiểm tra nếu có timeout
        has_timeout = summary.get('timeout_count', 0) > 0
        
        if summary["best_score"] is None:
            if has_timeout:
                f.write("Status          : Timeout (no solution found within time limit)\n")
            else:
                f.write("Status          : No solution found\n")
        else:
            f.write(f"Best score      : {summary['best_score']:.6f}\n")
            f.write(f"Mean score      : {summary['mean_score']:.6f}\n")
            f.write(f"Std score       : {summary['std_score']:.6f}\n")

        if has_timeout:
            f.write(f"Timeouts        : {summary['timeout_count']} / {summary['runs']} runs\n")

        f.write(f"Success rate    : {summary['success_rate']:.2%}\n")
        f.write(f"Mean time (s)   : {summary['mean_time']:.6e}\n")
        f.write(f"Mean effort     : {summary['mean_effort']:.2f}\n\n")
