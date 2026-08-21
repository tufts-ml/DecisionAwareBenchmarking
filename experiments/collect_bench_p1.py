"""
collect_bench_p1.py
-------------------
Collect Phase 1 (LR × Batch sweep) results and output the best (lr, batch)
config per method × task.

Reads all bench_p1_* prefixes and produces:
  1. A printed grid table (rows=methods, cols=tasks) showing best regret
  2. bench_p1_best.json — consumed by submit_bench_p2.sh to run Phase 2
     with the optimal configs per task.

Usage:
    python experiments/collect_bench_p1.py
    python experiments/collect_bench_p1.py --problem knapsack
    python experiments/collect_bench_p1.py --method mse
    python experiments/collect_bench_p1.py --relative   # show relative regret %
"""

import argparse
import csv
import json
import os

import numpy as np

# ---- Configuration (mirrors submit_bench_p1.sh) ----

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "pg_misspec", "shortestpath"]

PROB_ARG = {
    "knapsack":           "knapsack",
    "knapsack-real":      "knapsack",
    "energy":             "energy",
    "budgetalloc":        "budgetalloc",
    "cubic":              "cubic",
    "bipartitematching":  "bipartitematching",
    "portfolio":          "portfolio",
    "asurv":              "asurv",
    "cook_county":        "cook_county",
    "speed_humps":        "speed_humps",
    "sp_synth":           "sp_synth",
    "sp_planted":         "sp_planted",
    "pg_misspec":         "pg_misspec",
    "shortestpath":       "shortestpath",
}

PROB_VERSION = {
    "knapsack":           "gen",
    "knapsack-real":      "energy",
    "energy":             "energy",
    "budgetalloc":        "real",
    "cubic":              "gen",
    "bipartitematching":  "cora",
    "portfolio":          "real",
    "asurv":              "real",
    "cook_county":        "real",
    "speed_humps":        "real",
    "sp_synth":           "synth",
    "sp_planted":         "planted",
    "pg_misspec":         "v3",
    "shortestpath":       "warcraft",
}

METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce", "blackbox",
           "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
           "qptl", "cpLayer", "dad"]

# Methods only valid for subset of problems
METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      {"knapsack", "knapsack-real", "energy", "budgetalloc", "cubic",
                "bipartitematching", "portfolio", "asurv", "cook_county",
                "speed_humps", "sp_synth", "sp_planted", "pg_misspec"},
}

LRS = ["1e-2", "5e-3", "1e-3", "5e-2", "1e-1"]
BATCH_LABELS = ["default", "alt"]

# Multi-seed expansion (2026-05-21): seed 0 = original prefix; seeds 1..9 add
# `_s{N}` suffix. Cells excluded from the seed expansion (dad / energy /
# shortestpath / perturb×budgetalloc) only ever have seed 0 — aggregation
# degrades to single-seed for those.
SEEDS = list(range(10))

SEED_EXCLUDED_METHODS = {"dad"}
SEED_EXCLUDED_PROBLEMS = {"energy", "shortestpath"}
SEED_EXCLUDED_PAIRS = {("perturb", "budgetalloc")}

# Batch arms withdrawn from selection for a given (method, problem).
#
# listLTR x shortestpath: the full-batch ("default") arm is bs=1 on 10K warcraft
# images, i.e. 10K optimiser steps per epoch. Every default-batch cell was
# walltime-killed long before `--n_epochs 300` -- the val-best one died at epoch
# 54/300 and never reached ExpManager's final eval, so it has no results.npy and
# no test number. The task is therefore scored on the minibatch (`alt`,
# sgd bs=32) arm only, where lr=1e-3 ran the full 300 epochs.
BATCH_EXCLUDED_PAIRS = {
    ("listLTR", "shortestpath"): {"default"},
}

# Problems reported as absolute regret (not relative)
USE_ABSOLUTE = {"portfolio"}

RESULTS_ROOT = "saved_records"
BEST_JSON_PATH_TEST = "results/bench_p1_best.json"
BEST_JSON_PATH_VAL  = "results/bench_p1_best_val.json"


def seeds_for(prob, method):
    """Expected seed list for a (prob, method) cell.

    Seed-expanded cells get all 10 seeds; excluded cells only ever have seed 0.
    Used both for path-building and for completion accounting.
    """
    if method in SEED_EXCLUDED_METHODS: return [0]
    if prob in SEED_EXCLUDED_PROBLEMS: return [0]
    if (method, prob) in SEED_EXCLUDED_PAIRS: return [0]
    return list(SEEDS)


def batches_for(prob, method):
    """Batch arms this (prob, method) cell is selected over.

    Normally both arms; see BATCH_EXCLUDED_PAIRS for cells where one arm is
    withdrawn because it never produced usable runs.
    """
    dropped = BATCH_EXCLUDED_PAIRS.get((method, prob), set())
    return [b for b in BATCH_LABELS if b not in dropped]


def run_dir(prob, method, batch, lr, seed=0):
    out_dir = f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}"
    prefix = f"bench_p1_{method}_{batch}_lr{lr}"
    if seed > 0:
        prefix = f"{prefix}_s{seed}"
    return os.path.join(RESULTS_ROOT, out_dir, method, prefix)


def results_path(prob, method, batch, lr, seed=0):
    return os.path.join(run_dir(prob, method, batch, lr, seed), "results.npy")


def val_log_path(prob, method, batch, lr, seed=0):
    return os.path.join(run_dir(prob, method, batch, lr, seed), "val_logs.csv")


def train_log_txt_path(prob, method, batch, lr, seed=0):
    return os.path.join(run_dir(prob, method, batch, lr, seed), "log.txt")


def train_log_csv_path(prob, method, batch, lr, seed=0):
    return os.path.join(run_dir(prob, method, batch, lr, seed), "train_logs.csv")


def load_test_regret(path, absolute=False):
    """Return (abs_regret, rel_regret) from results.npy, or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        r = np.load(path, allow_pickle=True)
        regret = np.array(r[1], dtype=float)
        opt    = np.array(r[0], dtype=float)
        abs_r  = float(np.mean(regret))
        mean_opt = float(np.mean(np.abs(opt)))
        rel_r  = abs_r / mean_opt if mean_opt > 0 else None
        return abs_r, rel_r
    except Exception:
        return None, None


def load_val_regret(path):
    """
    Return (min_val_regret, best_epoch_label, source) from val_logs.csv, or
    (None, None, None). Uses the 'eval' column (per-epoch val decision regret).
    All 13 benchmark problems use regret (sense=1, lower is better).
    """
    if not os.path.exists(path):
        return None, None, None
    try:
        min_val = None
        best_epoch = None
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "eval" not in (reader.fieldnames or []):
                return None, None, None
            for row in reader:
                try:
                    v = float(row["eval"])
                except (TypeError, ValueError):
                    continue
                if min_val is None or v < min_val:
                    min_val = v
                    best_epoch = row.get("epoch")
        if min_val is None:
            return None, None, None
        return min_val, best_epoch, "val_eval"
    except Exception:
        return None, None, None


# Regex picks up lines like "Iter 42, val MSE (no solver): 5.23"
_VAL_MSE_LINE = __import__("re").compile(
    r"Iter\s+(\d+),\s*val MSE \(no solver\):\s*([0-9eE+\-\.]+)"
)

# Regex picks up lines like "Iter 42, train MSE (no solver): 5.23" (mse_train pseudo-method)
_TRAIN_MSE_LINE = __import__("re").compile(
    r"Iter\s+(\d+),\s*train MSE \(no solver\):\s*([0-9eE+\-\.]+)"
)


def load_train_pred_mse_from_log(path):
    """
    Parse log.txt for 'train MSE (no solver)' lines (mse_train pseudo-method)
    and return (min_train_mse, best_iter, source). Used when train_logs.csv
    is missing or malformed.
    """
    if not os.path.exists(path):
        return None, None, None
    try:
        min_val = None
        best_iter = None
        with open(path) as f:
            for line in f:
                m = _TRAIN_MSE_LINE.search(line)
                if not m:
                    continue
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                if min_val is None or v < min_val:
                    min_val = v
                    best_iter = f"Tr-{m.group(1)}"
        if min_val is None:
            return None, None, None
        return min_val, best_iter, "train_pred_mse"
    except Exception:
        return None, None, None


def load_train_mse_from_csv(path):
    """
    For mse_train runs we write train MSE into train_logs.csv's 'eval' column
    every epoch. Return (min_train_mse, best_epoch, source) or (None, None, None).
    """
    if not os.path.exists(path):
        return None, None, None
    try:
        min_val = None
        best_epoch = None
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "eval" not in (reader.fieldnames or []):
                return None, None, None
            for row in reader:
                try:
                    v = float(row["eval"])
                except (TypeError, ValueError):
                    continue
                if min_val is None or v < min_val:
                    min_val = v
                    best_epoch = row.get("epoch")
        if min_val is None:
            return None, None, None
        return min_val, best_epoch, "train_pred_mse_csv"
    except Exception:
        return None, None, None


def load_val_pred_mse_from_log(path):
    """
    Fallback for runs trained with --skip_solver_eval (no val_logs.csv, no
    solver-based val eval). Parse log.txt for 'val MSE (no solver)' lines and
    return (min_val_mse, best_iter, source). These runs used val pred MSE for
    early stopping, so that is the metric that was actually used to pick the
    best checkpoint — still non-leaky.
    """
    if not os.path.exists(path):
        return None, None, None
    try:
        min_val = None
        best_iter = None
        with open(path) as f:
            for line in f:
                m = _VAL_MSE_LINE.search(line)
                if not m:
                    continue
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                if min_val is None or v < min_val:
                    min_val = v
                    best_iter = f"Tr-{m.group(1)}"
        if min_val is None:
            return None, None, None
        return min_val, best_iter, "val_pred_mse"
    except Exception:
        return None, None, None


def load_signal(prob, method, batch, lr, seed, metric):
    """Load the selection signal for one (prob, method, batch, lr, seed) cell.

    Returns (value, extras_dict) where value is None if the cell is incomplete.
    extras_dict carries diagnostics (best_epoch, val_source) for the val branch.
    """
    if metric == "test":
        absolute = prob in USE_ABSOLUTE
        abs_r, rel_r = load_test_regret(results_path(prob, method, batch, lr, seed), absolute)
        v = abs_r if absolute else rel_r
        return v, {}
    if metric != "val":
        raise ValueError(f"unknown metric: {metric}")

    # mse_train / mse_val pseudo-methods route to method-specific signal readers.
    if method == "mse_train":
        v, best_epoch, source = load_train_mse_from_csv(
            train_log_csv_path(prob, method, batch, lr, seed))
        if v is None:
            v, best_epoch, source = load_train_pred_mse_from_log(
                train_log_txt_path(prob, method, batch, lr, seed))
    elif method == "mse_val":
        # No val_logs.csv when solver_valfreq==0; parse log.txt.
        v, best_epoch, source = load_val_pred_mse_from_log(
            train_log_txt_path(prob, method, batch, lr, seed))
    else:
        v, best_epoch, source = load_val_regret(val_log_path(prob, method, batch, lr, seed))
        if v is None:
            # Fallback for --skip_solver_eval runs (e.g. energy/mse)
            v, best_epoch, source = load_val_pred_mse_from_log(
                train_log_txt_path(prob, method, batch, lr, seed))
    return v, {"best_epoch": best_epoch, "val_source": source}


def best_config_for(prob, method, metric="test"):
    """
    Return (best_summary, best_batch, best_lr, n_complete_seeds_at_best,
            n_total_seeds_at_best, extras).

    best_summary is the mean of the selection signal across all completed seeds
    at the winning (batch, lr). The selection is performed on this mean, so the
    winner is the multi-seed mean-best — not a single-seed best.

    extras includes per-seed values, std, and the seed list at the winning cell.
    Cells excluded from the seed expansion (dad, energy, shortestpath,
    perturb×budgetalloc) only have seed 0 → aggregation degrades to one value.
    """
    expected_seeds = seeds_for(prob, method)
    # Each (batch, lr) cell aggregates across its expected seed list.
    cell_means = {}      # (batch, lr) -> mean across completed seeds (None if 0 done)
    cell_records = {}    # (batch, lr) -> {seed_values, mean, std, n_done, n_total, per_seed_extras}
    for batch in batches_for(prob, method):
        for lr in LRS:
            per_seed_vals = {}
            per_seed_extras = {}
            for seed in expected_seeds:
                v, ex = load_signal(prob, method, batch, lr, seed, metric)
                if v is not None:
                    per_seed_vals[seed] = v
                    per_seed_extras[seed] = ex
            n_done = len(per_seed_vals)
            n_total = len(expected_seeds)
            if n_done == 0:
                cell_means[(batch, lr)] = None
                cell_records[(batch, lr)] = {
                    "seed_values": {}, "mean": None, "std": None,
                    "n_done": 0, "n_total": n_total, "per_seed_extras": {},
                }
                continue
            vals_arr = np.array(list(per_seed_vals.values()), dtype=float)
            mean = float(vals_arr.mean())
            std  = float(vals_arr.std(ddof=1)) if n_done > 1 else 0.0
            cell_means[(batch, lr)] = mean
            cell_records[(batch, lr)] = {
                "seed_values": {int(s): float(v) for s, v in per_seed_vals.items()},
                "mean": mean, "std": std,
                "n_done": n_done, "n_total": n_total,
                "per_seed_extras": per_seed_extras,
            }

    completed = {k: v for k, v in cell_means.items() if v is not None}
    if not completed:
        return None, None, None, 0, len(expected_seeds), {}

    best_key = min(completed, key=lambda k: completed[k])
    best_rec = cell_records[best_key]
    extras = {
        "seed_values": best_rec["seed_values"],
        "std": best_rec["std"],
    }
    # Carry through best_epoch / val_source from the seed-0 run if available
    # (or any completed seed if seed 0 missing) for backward compat with the
    # existing JSON consumers.
    seed0_ex = best_rec["per_seed_extras"].get(0)
    if seed0_ex is None and best_rec["per_seed_extras"]:
        first_seed = sorted(best_rec["per_seed_extras"].keys())[0]
        seed0_ex = best_rec["per_seed_extras"][first_seed]
    if seed0_ex:
        extras.update({k: v for k, v in seed0_ex.items() if v is not None})

    return (best_rec["mean"], best_key[0], best_key[1],
            best_rec["n_done"], best_rec["n_total"], extras)


def print_results(args):
    problems  = [args.problem] if args.problem else PROBLEMS
    methods   = [args.method]  if args.method  else METHODS
    use_pct   = args.relative
    metric    = args.metric

    # Column widths
    COL_W = 12
    METH_W = 12

    header_cells = [f"{p[:COL_W]:>{COL_W}}" for p in problems]
    print(f"\n  {'method':>{METH_W}}  " + "  ".join(header_cells))
    print("  " + "-" * (METH_W + 2 + (COL_W + 2) * len(problems)))

    # For building bench_p1_best{,_val}.json
    best_json = {}

    for method in methods:
        row = []
        for prob in problems:
            # Check if method applies to this problem
            allowed = METHOD_PROBLEMS.get(method, None)
            if allowed is not None and prob not in allowed:
                row.append(f"{'N/A':>{COL_W}}")
                continue

            best_val, best_batch, best_lr, n_done, n_total, extras = best_config_for(
                prob, method, metric=metric)

            if best_val is None:
                row.append(f"{'—':>{COL_W}}")
                continue

            absolute = prob in USE_ABSOLUTE
            # --relative only meaningful for metric=test (rel vs abs). For val
            # selection, we always display the absolute val regret.
            if metric == "test" and use_pct and not absolute:
                s = f"{best_val*100:.2f}%({n_done}/{n_total})"
            else:
                s = f"{best_val:.4f}({n_done}/{n_total})"
            row.append(f"{s:>{COL_W}}")

            # Record best config. Key "val" retained so existing readers
            # (submit_bench_p2.sh) keep working — it only reads lr/batch.
            # n_done / n_total now count SEEDS at the winning (batch, lr).
            entry = {
                "lr": best_lr,
                "batch": best_batch,
                "val": round(best_val, 6),
                "n_done": n_done,
                "n_total": n_total,
                "metric": metric,
            }
            if extras.get("std") is not None:
                entry["std"] = round(extras["std"], 6)
            if extras.get("seed_values"):
                entry["seed_values"] = extras["seed_values"]
            if metric == "val":
                if extras.get("best_epoch") is not None:
                    entry["best_epoch"] = extras["best_epoch"]
                if extras.get("val_source") is not None:
                    entry["val_source"] = extras["val_source"]
            best_json.setdefault(method, {})[prob] = entry

        print(f"  {method:>{METH_W}}  " + "  ".join(row))

    print()
    return best_json


def main():
    parser = argparse.ArgumentParser(description="Collect Phase 1 sweep results")
    parser.add_argument("--problem", type=str, default=None,
                        help="Filter to one problem")
    parser.add_argument("--method", type=str, default=None,
                        help="Filter to one method")
    parser.add_argument("--relative", action="store_true",
                        help="Show relative regret (regret/opt) as %%")
    parser.add_argument("--metric", type=str, choices=["test", "val"], default="val",
                        help="Selection metric. 'val' (default) = correct "
                             "(picks by min val regret from val_logs.csv). "
                             "'test' = legacy (LEAKY: picks by test regret "
                             "from results.npy); kept only for diffing.")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip writing the best-config JSON file")
    args = parser.parse_args()

    out_path = BEST_JSON_PATH_VAL if args.metric == "val" else BEST_JSON_PATH_TEST

    print("=" * 70)
    print(f"Phase 1 results — best (lr, batch) per method × task (metric={args.metric})")
    if args.metric == "test":
        print("  ⚠ LEAKY: selecting by test regret (results.npy[1]). Use --metric val.")
    print("=" * 70)

    best_json = print_results(args)

    if not args.no_json and best_json:
        with open(out_path, "w") as f:
            json.dump(best_json, f, indent=2)
        print(f"Best configs written to {out_path}")
        if args.metric == "val":
            print("To drive Phase 2 from val-selected configs, point "
                  "submit_bench_p2.sh at this file (BEST_JSON).")
        else:
            print("Pass this to Phase 2: bash slurm/submit_bench_p2.sh")

    # Print completion summary (counts every (prob, method, batch, lr, seed)
    # cell, where seed list is determined by seeds_for()).
    n_done = n_total = 0
    for prob in PROBLEMS:
        for method in METHODS:
            allowed = METHOD_PROBLEMS.get(method, None)
            if allowed is not None and prob not in allowed:
                continue
            for batch in batches_for(prob, method):
                for lr in LRS:
                    for seed in seeds_for(prob, method):
                        n_total += 1
                        if os.path.exists(results_path(prob, method, batch, lr, seed)):
                            n_done += 1
    print(f"\nOverall completion: {n_done}/{n_total} jobs done (seed-aware)")


if __name__ == "__main__":
    main()
