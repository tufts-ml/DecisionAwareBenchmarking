"""
eval_test_regret.py
-------------------
Offline pass: recompute the **test decision regret** for a run directory from
its saved prediction-model checkpoint (`checkpoints/tr_pred_best.pt`), without
re-training.

This is the decision-side twin of `eval_test_pred.py`. It exists for cells
whose training job was walltime-killed before `ExpManager` reached its final
`print_metrics` call — those runs have `train_logs.csv` / `val_logs.csv` /
checkpoints but **no `results.npy`**, so `collect_loss_matrix.py` reports
their test regret as missing.

It replicates exactly what `ExpManager.run` does at the end of training:

    Z_test_opt, _ = problem.get_decision(Y_test, ..., isTrain=False)
    Zs_hat,    _ = problem.get_decision(model(X_test), ..., isTrain=False)
    regret       = get_eval_results(problem, Y_test, Z_test_opt, Zs_hat, Y_aux)

The result is written to `{run_dir}/test_regret_offline.json` — **not** to
`results.npy`, so an incomplete run is never mistaken for a completed one.
Downstream scripts must opt in to reading it.

Caveat: the checkpoint is the best-val model *as of the epoch the job died*,
not of a full `--n_epochs` run. Numbers produced here are "best available",
not protocol-identical to cells that trained to convergence/patience. Say so
wherever they are reported.

Usage:
    python experiments/eval_test_regret.py --run_dir saved_records/.../prefix \
                                           --problem shortestpath
    python experiments/eval_test_regret.py --missing        # every gap in loss_matrix.json
    python experiments/eval_test_regret.py --missing --force
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

os.environ.setdefault("PYTHONHASHSEED", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_test_pred import PROB_CONFIG, make_args  # noqa: E402

from openpto.config.utils_conf import load_conf  # noqa: E402
from openpto.method.Predicts.wrapper_predicts import pred_model_wrapper  # noqa: E402
from openpto.method.Solvers.wrapper_solver import solver_wrapper  # noqa: E402
from openpto.method.utils_method import to_array, to_device  # noqa: E402
from openpto.metrics.evals import get_eval_results  # noqa: E402
from openpto.problems.wrapper_prob import problem_wrapper  # noqa: E402

# Solver used per problem in the Phase-1/2 sweep (mirrors CLAUDE.md's
# "Solvers for comparability" table and slurm/submit_bench_p1.sh).
PROB_SOLVER = {
    "knapsack": "heuristic",
    "knapsack-real": "gurobi",
    "energy": "gurobi",
    "budgetalloc": "neural",
    "cubic": "heuristic",
    "bipartitematching": "cvxpy",
    "portfolio": "cvxpy",
    "asurv": "heuristic",
    "cook_county": "heuristic",
    "speed_humps": "heuristic",
    "sp_synth": "heuristic",
    "sp_planted": "heuristic",
    "pg_misspec": "heuristic",
    "shortestpath": "heuristic",
}

# (problem, solver) -> (problem, args, conf, ptoSolver, test-optimal cache)
_CACHE = {}


def get_context(prob):
    """Build (problem, args, conf, ptoSolver) once per problem, and solve the
    test set at the true Y to get the optimal decisions."""
    if prob in _CACHE:
        return _CACHE[prob]

    args = make_args(prob)
    args.solver = PROB_SOLVER[prob]
    conf = load_conf(
        prob_path=PROB_CONFIG[prob],
        method_path="openpto/config/models/default.yaml",
        prob_name=args.problem,
    )
    problem = problem_wrapper(args, conf)
    ptoSolver = solver_wrapper(args, conf, problem)

    X_test, Y_test, Y_test_aux = problem.get_test_data()
    Z_test_opt, Objs_test_opt = problem.get_decision(
        Y_test,
        params=Y_test_aux,
        ptoSolver=ptoSolver,
        isTrain=False,
        **problem.init_API(),
    )
    Objs_test_opt = to_array(Objs_test_opt)

    ctx = dict(
        problem=problem, args=args, conf=conf, ptoSolver=ptoSolver,
        X_test=X_test, Y_test=Y_test, Y_test_aux=Y_test_aux,
        Z_test_opt=Z_test_opt, Objs_test_opt=Objs_test_opt,
    )
    _CACHE[prob] = ctx
    return ctx


@torch.no_grad()
def compute_test_regret(prob, ckpt_path):
    """Return (regret_per_instance, Objs_test_opt) as numpy arrays."""
    ctx = get_context(prob)
    problem, args = ctx["problem"], ctx["args"]

    pred_model_args = {
        "ipdim": problem.get_model_shape()[0],
        "opdim": problem.get_model_shape()[1],
        "out_act": problem.get_output_activation(),
    }
    model = pred_model_wrapper(args, pred_model_args)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    preds = model(ctx["X_test"])
    Zs_hat, _ = problem.get_decision(
        to_device(preds, "cpu"),
        params=ctx["Y_test_aux"],
        ptoSolver=ctx["ptoSolver"],
        isTrain=False,
        **problem.init_API(),
    )
    eval_result = get_eval_results(
        problem, ctx["Y_test"], ctx["Z_test_opt"], Zs_hat, ctx["Y_test_aux"]
    )
    return to_array(eval_result["value"]), ctx["Objs_test_opt"]


def write_result(run_dir, prob, regret, objs_opt):
    abs_r = float(np.mean(regret))
    mean_opt = float(np.mean(np.abs(objs_opt)))
    rel_r = abs_r / mean_opt if mean_opt > 0 else float("nan")
    payload = {
        "problem": prob,
        "test_regret_abs": abs_r,
        "test_regret_rel": rel_r,
        "mean_optimal_objective": mean_opt,
        "n_test": int(np.size(regret)),
        "source": "eval_test_regret.py (offline, from checkpoints/tr_pred_best.pt)",
        "caveat": "run did not reach ExpManager's final eval; checkpoint is the "
                  "best-val model as of the last completed epoch",
    }
    out = os.path.join(run_dir, "test_regret_offline.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out, payload


def iter_missing(loss_matrix_path):
    """Yield (prob, method, run_dir) for cells with no test regret."""
    with open(loss_matrix_path) as f:
        lm = json.load(f)
    for prob in lm:
        for method, entry in lm[prob].items():
            if entry["metrics"].get("test_regret_abs") is None:
                yield prob, method, entry["run_dir"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--problem", default=None)
    ap.add_argument("--missing", action="store_true",
                    help="Process every cell in loss_matrix.json whose test "
                         "regret is missing.")
    ap.add_argument("--loss_matrix_json", default="loss_matrix.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.missing:
        targets = list(iter_missing(args.loss_matrix_json))
    elif args.run_dir:
        if not args.problem:
            ap.error("--run_dir requires --problem")
        targets = [(args.problem, os.path.basename(os.path.dirname(args.run_dir)),
                    args.run_dir)]
    else:
        ap.error("pass --run_dir/--problem or --missing")

    if not targets:
        print("nothing to do — no missing test-regret cells")
        return

    n_done = n_skip = n_fail = 0
    for prob, method, run_dir in targets:
        ckpt = os.path.join(run_dir, "checkpoints", "tr_pred_best.pt")
        out_path = os.path.join(run_dir, "test_regret_offline.json")
        if not os.path.exists(ckpt):
            print(f"  [skip] {prob}/{method}: no tr_pred_best.pt")
            n_skip += 1
            continue
        if os.path.exists(out_path) and not args.force:
            print(f"  [skip] {prob}/{method}: already done (--force to redo)")
            n_skip += 1
            continue
        try:
            regret, objs_opt = compute_test_regret(prob, ckpt)
        except Exception as e:
            print(f"  [FAIL] {prob}/{method}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        out, payload = write_result(run_dir, prob, regret, objs_opt)
        print(f"  {prob:18s} {method:10s}  abs={payload['test_regret_abs']:.6f}  "
              f"rel={payload['test_regret_rel']:.6f}  n={payload['n_test']}  -> {out}")
        n_done += 1

    print(f"\ndone: {n_done}  skipped: {n_skip}  failed: {n_fail}")


if __name__ == "__main__":
    main()
