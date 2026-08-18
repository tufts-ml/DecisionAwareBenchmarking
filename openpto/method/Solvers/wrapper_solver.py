from importlib import import_module

################################# Wrappers ################################################
# Solver classes are imported lazily so that a run only pays the import cost
# (and dependency requirements, e.g. gurobipy / cvxpy) of the backend it uses.
_SOLVER_REGISTRY = {
    "asurv": {"heuristic": ("heuristic.TopKSolver", "TopKSolver")},
    "cook_county": {"heuristic": ("heuristic.TopKSolver", "TopKSolver")},
    "speed_humps": {"heuristic": ("heuristic.TopKSolver", "TopKSolver")},
    "budgetalloc": {"neural": ("neural.BudgetallocSolver", "budgetallocSolver")},
    "bipartitematching": {"cvxpy": ("cvxpy.cp_bmatching", "BmatchingSolver")},
    "portfolio": {"cvxpy": ("cvxpy.cp_port", "CpPortfolioSolver")},
    "cubic": {
        "heuristic": ("heuristic.TopKSolver", "TopKSolver"),
        "neural": ("neural.softTopkSolver", "softTopkSolver"),
    },
    "energy": {"gurobi": ("grb.grb_energy", "ICONGrbSolver")},
    "knapsack": {
        "gurobi": ("grb.grb_knapsack", "KPGrbSolver"),
        "heuristic": ("heuristic.dp", "DPSolver"),
        "qptl": ("grb.grb_qpsolver", "QPGrbSolver"),
        "cvxpy": ("cvxpy.cp_kp", "CpKPSolver"),
    },
    "shortestpath": {"heuristic": ("heuristic.spSolver", "spSolver")},
    "sp_synth": {"heuristic": ("heuristic.dagSPSolver", "dagSPSolver")},
    "sp_planted": {"heuristic": ("heuristic.dagSPSolver", "dagSPSolver")},
    "pg_misspec": {"heuristic": ("heuristic.BinarySignSolver", "BinarySignSolver")},
}


def solver_wrapper(args, conf, problem):
    module_name, class_name = _SOLVER_REGISTRY[args.problem][args.solver]
    module = import_module(f"openpto.method.Solvers.{module_name}")
    SolverClass = getattr(module, class_name)
    solve_dict = {**problem.init_API(), **conf["solver"][args.solver]}
    return SolverClass(**solve_dict)
