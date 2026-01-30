from problems.discrete.grid_pathfinding import GridPathfinding

def make_problem(problem_cfg):
    name = problem_cfg["name"]
    params = problem_cfg["params"]

    if name == "grid_pathfinding":
        return GridPathfinding(
            grid=params["grid"],
            start=tuple(params["start"]),
            goal=tuple(params["goal"]),
            diagonal=params.get("diagonal", False)
        )

    raise ValueError(f"Unknown problem: {name}")
