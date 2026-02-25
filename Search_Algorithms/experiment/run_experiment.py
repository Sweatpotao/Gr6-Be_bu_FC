import yaml
from .registry import PROBLEM_REGISTRY, ALGORITHM_REGISTRY
from experiment.experiment_runner import ExperimentRunner

def run_experiment(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ---- Init problem from config ----
    prob_cfg = config["problem"]
    problem_name = prob_cfg["name"]

    ProblemClass = PROBLEM_REGISTRY[problem_name]
    problem = ProblemClass(**prob_cfg["params"])

    runs = config["experiment"]["runs"]
    results = {}

    # ---- Loop algorithms ----
    for algo_name, algo_cfg in config["algorithms"].items():
        AlgoClass = ALGORITHM_REGISTRY[algo_name]

        runner = ExperimentRunner(
            AlgoClass,
            problem,
            algo_cfg,
            runs=runs
        )

        print(f"Running {algo_name} on {problem_name}")
        results[algo_name] = runner.run()

    return results


def run_experiment_from_dict(config):
    """Run experiment using config dict directly (no YAML file needed).
    
    Args:
        config: Dict with structure:
            {
                "problem": {"name": str, "params": dict},
                "algorithms": {"AlgoName": {"timeout": int, ...}, ...},
                "experiment": {"runs": int}
            }
    
    Returns:
        Dict of results per algorithm
    """
    # ---- Init problem from config ----
    prob_cfg = config["problem"]
    problem_name = prob_cfg["name"]

    ProblemClass = PROBLEM_REGISTRY[problem_name]
    problem = ProblemClass(**prob_cfg["params"])

    runs = config["experiment"]["runs"]
    results = {}

    # ---- Loop algorithms ----
    for algo_name, algo_cfg in config["algorithms"].items():
        AlgoClass = ALGORITHM_REGISTRY[algo_name]

        runner = ExperimentRunner(
            AlgoClass,
            problem,
            algo_cfg,
            runs=runs
        )

        print(f"Running {algo_name} on {problem_name}")
        results[algo_name] = runner.run()

    return results
