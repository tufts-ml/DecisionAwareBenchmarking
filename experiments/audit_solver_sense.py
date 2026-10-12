"""
audit_solver_sense.py
---------------------
For every benchmark problem, check that the solver used in the sweep optimises
``problem.get_objective`` in the direction ``init_API()['modelSense']`` declares.
Decisions from perturbed / shuffled / negated costs are scored under the TRUE
costs and compared with the solver's own decision: if no candidate beats it the
solver MAXIMISES, if none is worse it MINIMISES.

This is the check behind the Cubic Top-K sense correction (REPRODUCING.md,
"Cubic Top-K sense correction"): ``cubic`` was the only task whose declaration
(MINIMIZE, inherited from upstream) contradicted its solver (``torch.topk``,
i.e. MAXIMIZE). After the fix every task's declared sense matches its solver.

Portfolio is reported UNCLEAR by this heuristic for a different reason:
``PortfolioOpt.get_objective`` expects the Cholesky factor of the covariance as
``aux_data`` (it takes ``cholesky()`` itself when aux is None and
``get_decision`` passes ``sqrt_covar``), but the evaluation path hands it the
raw covariance (``Y_*_aux = covar_mat[...]``), so the scored risk term is
``||Sigma z||^2`` rather than the ``||L z||^2`` the solver optimised. Under the
solver's own objective the solver is near-optimal; under the scored one random
candidates can beat it. This is inherited from upstream (``PortfolioOpt.py`` and
``openpto/metrics/evals.py`` are unmodified) and affects every method equally.

Usage (Tier 2 tooling: needs the datasets under openpto/data/; ``energy`` needs
a Gurobi licence, ``shortestpath`` the Warcraft data):
    python experiments/audit_solver_sense.py                 # all problems
    python experiments/audit_solver_sense.py cubic knapsack  # a subset
    DETAIL=1 python experiments/audit_solver_sense.py portfolio
"""
import os
import sys
import time

# Load the env's own libstdc++ before torch (see experiments/main_results.py).
import ctypes

_libstdcxx = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
if os.path.exists(_libstdcxx):
    ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gurobipy import GRB  # noqa: E402  pylint: disable=no-name-in-module

import eval_test_regret as etr  # noqa: E402
from openpto.method.utils_method import to_tensor  # noqa: E402

_pw = etr.problem_wrapper


def _pw2(a, c):
    pr = _pw(a, c)
    if not hasattr(pr, "device"):
        pr.device = torch.device("cpu")
    return pr


etr.problem_wrapper = _pw2

ALL_PROBLEMS = [
    "cubic", "knapsack", "asurv", "cook_county", "speed_humps", "sp_synth", "sp_planted",
    "pg_misspec", "budgetalloc", "portfolio", "knapsack-real", "bipartitematching",
    "energy", "shortestpath",
]


def audit(prob, n_instances=6, detail=False):
    """Return (declared, actual, verdict, stats) for one problem."""
    ctx = etr.get_context(prob)
    problem, solver = ctx["problem"], ctx["ptoSolver"]
    if not hasattr(problem, "device"):
        problem.device = torch.device("cpu")
    Y = to_tensor(ctx["Y_test"])
    aux = ctx["Y_test_aux"]
    n = min(n_instances, Y.shape[0])
    Y = Y[:n]
    aux_n = (
        aux[:n]
        if hasattr(aux, "__len__") and not isinstance(aux, dict) and len(aux) >= n
        else aux
    )
    declared = {GRB.MAXIMIZE: "MAXIMIZE", GRB.MINIMIZE: "MINIMIZE"}[
        problem.init_API()["modelSense"]
    ]

    def obj_of(Yc):
        if prob == "portfolio":
            Z, _ = problem.get_decision(
                Yc, params=aux_n, aux_data=aux_n, ptoSolver=solver, isTrain=False,
                **problem.init_API(),
            )
        else:
            Z, _ = problem.get_decision(
                Yc, params=aux_n, ptoSolver=solver, isTrain=False, **problem.init_API()
            )
        return to_tensor(problem.get_objective(Y, to_tensor(Z), aux_n)).detach().float().flatten()[:n]

    base = obj_of(Y)
    cands = []
    for _ in range(3):  # per-instance shuffle of the cost entries along the item axis
        idx = torch.stack([torch.randperm(Y.shape[1]) for _ in range(n)])
        Yp = torch.stack([Y[i][idx[i]] for i in range(n)])
        cands.append(obj_of(Yp))
    for _ in range(3):  # multiplicative noise (keeps sign)
        cands.append(obj_of(Y * (1 + 0.7 * torch.randn_like(Y)).clamp(min=0.05)))
    try:
        cands.append(obj_of(-Y))
    except Exception:  # some solvers reject negated costs
        pass
    C = torch.stack(cands)  # [n_cand, n]
    if detail:
        print("   per-candidate (cand - obj(z*)), rows=candidates:",
              (C - base).numpy().round(4), flush=True)
    gt = int((C > base + 1e-6).sum())
    lt = int((C < base - 1e-6).sum())
    if lt > 0 and gt == 0:
        actual = "MAXIMIZE"
    elif gt > 0 and lt == 0:
        actual = "MINIMIZE"
    else:
        actual = "UNCLEAR"
    verdict = "OK" if actual == declared else ("MISMATCH" if actual != "UNCLEAR" else "?")
    stats = dict(base=float(base.mean()), best=float(C.max(0).values.mean()),
                 worst=float(C.min(0).values.mean()), gt=gt, lt=lt)
    return declared, actual, verdict, stats


def main():
    torch.manual_seed(0)
    order = sys.argv[1:] or ALL_PROBLEMS
    detail = bool(os.environ.get("DETAIL"))
    print(f"{'problem':18s} {'solver':9s} {'declared':9s} | {'obj(z*)':>9} {'best cand':>9} "
          f"{'worst cand':>10} {'#cand>z*':>8} {'#cand<z*':>8} | verdict", flush=True)
    n_bad = 0
    for prob in order:
        t0 = time.time()
        try:
            declared, actual, verdict, s = audit(prob, detail=detail)
            n_bad += verdict == "MISMATCH"
            print(f"{prob:18s} {etr.PROB_SOLVER[prob]:9s} {declared:9s} | {s['base']:9.4g} "
                  f"{s['best']:9.4g} {s['worst']:10.4g} {s['gt']:8d} {s['lt']:8d} | "
                  f"solver {actual} -> {verdict}   [{time.time() - t0:.0f}s]", flush=True)
        except Exception as e:  # keep going: a missing dataset/licence should not hide the rest
            print(f"{prob:18s} ERROR {type(e).__name__}: {str(e)[:120]}   "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
