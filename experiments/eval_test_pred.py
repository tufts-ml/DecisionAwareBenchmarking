"""
eval_test_pred.py
-----------------
Offline pass: for each (problem, method) best-config run directory listed in
loss_matrix.json, load the saved prediction model checkpoint
(`checkpoints/tr_pred_best.pt`), forward `X_{train,val,test}` through it, and
compute the two-stage prediction loss (MSE / BCE / CE / MAE — per the
problem's `get_twostageloss()`). Write the three numbers to
`{run_dir}/test_pred_loss.json`.

This fills in the only metric that isn't already on disk in the Phase-1/2
sweep: the **test** prediction loss. `collect_loss_matrix.py` then picks it
up. No re-training.

Usage:
    python experiments/eval_test_pred.py                       # all problems
    python experiments/eval_test_pred.py --problem budgetalloc
    python experiments/eval_test_pred.py --force               # recompute
"""

import argparse
import json
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

os.environ.setdefault("PYTHONHASHSEED", "0")

from openpto.config.utils_conf import load_conf
from openpto.method.Models.utils_loss import str2twoStageLoss
from openpto.method.Predicts.wrapper_predicts import pred_model_wrapper
from openpto.problems.wrapper_prob import problem_wrapper


# ---- Per-problem configuration (mirrors slurm/submit_bench_p1.sh) ----

PROB_ARG = {
    "knapsack": "knapsack", "knapsack-real": "knapsack",
    "energy": "energy", "budgetalloc": "budgetalloc",
    "cubic": "cubic", "bipartitematching": "bipartitematching",
    "portfolio": "portfolio", "asurv": "asurv", "cook_county": "cook_county",
    "speed_humps": "speed_humps", "sp_synth": "sp_synth",
    "sp_planted": "sp_planted", "pg_misspec": "pg_misspec",
    "shortestpath": "shortestpath",
}

INSTANCES = {
    "knapsack": 400, "knapsack-real": 400, "energy": 400,
    "budgetalloc": 400, "cubic": 250, "bipartitematching": 20,
    "portfolio": 400, "asurv": 400, "cook_county": 400,
    "speed_humps": 400, "sp_synth": 400, "sp_planted": 400,
    "pg_misspec": 400, "shortestpath": 10000,
}

TESTINSTANCES = {
    "knapsack": 200, "knapsack-real": 200, "energy": 200,
    "budgetalloc": 200, "cubic": 400, "bipartitematching": 6,
    "portfolio": 200, "asurv": 200, "cook_county": 200,
    "speed_humps": 200, "sp_synth": 10000, "sp_planted": 10000,
    "pg_misspec": 10000, "shortestpath": 1000,
}

PROB_CONFIG = {
    "knapsack": "openpto/config/probs/knapsack_small.yaml",
    "knapsack-real": "openpto/config/probs/knapsack-real.yaml",
    "energy": "",
    "budgetalloc": "",
    "cubic": "",
    "bipartitematching": "",
    "portfolio": "",
    "asurv": "openpto/config/probs/asurv.yaml",
    "cook_county": "openpto/config/probs/cook_county.yaml",
    "speed_humps": "openpto/config/probs/speed_humps.yaml",
    "sp_synth": "openpto/config/probs/sp_synth.yaml",
    "sp_planted": "openpto/config/probs/sp_planted.yaml",
    "pg_misspec": "openpto/config/probs/pg_misspec.yaml",
    "shortestpath": "openpto/config/probs/shortestpath.yaml",
}

# (pred_model, n_layers) override per problem; default (dense, 2)
PRED_MODEL_OVERRIDE = {
    "sp_synth":     ("dense", 1),
    "sp_planted":   ("dense", 1),
    "pg_misspec":   ("dense", 1),
    "shortestpath": ("Resnet18", 2),
}


def make_args(prob):
    """Minimal Namespace of args needed by problem_wrapper and pred_model_wrapper."""
    pred_model, n_layers = PRED_MODEL_OVERRIDE.get(prob, ("dense", 2))
    prob_arg = PROB_ARG[prob]

    ns = SimpleNamespace(
        problem=prob_arg,
        seed=2023,
        instances=INSTANCES[prob],
        testinstances=TESTINSTANCES[prob],
        val_frac=0.2,
        loadnew=False,
        data_dir=os.path.join("./openpto/data/", prob_arg),
        n_layers=n_layers,
        n_hidden=32,
        activation="relu",
        pred_model=pred_model,
    )
    return ns


# ---- Shared problem cache (so we pay init_if_not_saved once per problem) ----

_PROBLEM_CACHE = {}


def get_problem(prob):
    if prob in _PROBLEM_CACHE:
        return _PROBLEM_CACHE[prob]
    args = make_args(prob)
    conf = load_conf(
        prob_path=PROB_CONFIG[prob],
        method_path="openpto/config/models/default.yaml",
        prob_name=args.problem,
    )
    problem = problem_wrapper(args, conf)
    _PROBLEM_CACHE[prob] = (problem, args, conf)
    return _PROBLEM_CACHE[prob]


# ---- Main eval ----

@torch.no_grad()
def compute_pred_losses(prob, ckpt_path):
    """Return dict with train / val / test two-stage prediction loss."""
    problem, args, conf = get_problem(prob)

    pred_model_args = {
        "ipdim": problem.get_model_shape()[0],
        "opdim": problem.get_model_shape()[1],
        "out_act": problem.get_output_activation(),
    }
    model = pred_model_wrapper(args, pred_model_args)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    twostage = str2twoStageLoss(problem)

    X_train, Y_train, _ = problem.get_train_data()
    X_val,   Y_val,   _ = problem.get_val_data()
    X_test,  Y_test,  _ = problem.get_test_data()

    out = {}
    for name, X, Y in (("train", X_train, Y_train),
                        ("val",   X_val,   Y_val),
                        ("test",  X_test,  Y_test)):
        preds = model(X)
        loss = twostage(problem, preds, Y, reduction="mean")
        out[name] = float(loss.detach().cpu().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_matrix_json", default="results/loss_matrix.json")
    ap.add_argument("--problem", default=None, help="Restrict to one problem.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if test_pred_loss.json exists.")
    args = ap.parse_args()

    if not os.path.exists(args.loss_matrix_json):
        print(f"error: {args.loss_matrix_json} not found. "
              f"Run `python experiments/collect_loss_matrix.py` first.")
        sys.exit(1)

    with open(args.loss_matrix_json) as f:
        data = json.load(f)

    n_done = n_skipped = n_fail = 0
    for prob in sorted(data.keys()):
        if args.problem and prob != args.problem:
            continue
        for method in sorted(data[prob].keys()):
            entry = data[prob][method]
            run_dir = entry["run_dir"]
            ckpt = os.path.join(run_dir, "checkpoints", "tr_pred_best.pt")
            out_path = os.path.join(run_dir, "test_pred_loss.json")

            if not os.path.exists(ckpt):
                # no checkpoint; skip silently
                continue
            if os.path.exists(out_path) and not args.force:
                n_skipped += 1
                continue

            try:
                losses = compute_pred_losses(prob, ckpt)
            except Exception as e:
                print(f"  [FAIL] {prob}/{method}: {type(e).__name__}: {e}")
                n_fail += 1
                continue

            payload = {
                "train_pred_loss": losses["train"],
                "val_pred_loss":   losses["val"],
                "test_pred_loss":  losses["test"],
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            n_done += 1
            print(f"  {prob:18s} {method:10s}  "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}  test={losses['test']:.4f}")

    print(f"\ndone: {n_done}  skipped (already done): {n_skipped}  failed: {n_fail}")


if __name__ == "__main__":
    main()
