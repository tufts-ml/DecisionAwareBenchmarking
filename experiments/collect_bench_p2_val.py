"""
collect_bench_p2_val.py
-----------------------
Pick the val-best Phase-2 hyperparameter per (method, problem) and write
``bench_p2_best_val.json``.

For every (method, problem) in ``bench_p1_best_val.json``:
  * Phase-2-tunable methods (dfl, blackbox, qptl, listLTR, lodl, perturb, pg,
    dad): enumerate every prefix that the Phase-2 sweep produced, read its
    ``val_logs.csv``, take the min of the ``eval`` column, pick the HP value
    whose run had the lowest min val regret. ``perturb`` is treated as a
    single method with two sub-sweeps (sigma at n=10, n_samples at sigma=1.0)
    — all 10 candidates compete jointly.
  * Other methods (mse, identity, spo, nce, pointLTR, pairLTR, cpLayer):
    no method-specific HP sweep, so the winner is the Phase-1 best entry.

For each winner we also record:
  * ``test_regret`` — read from ``results.npy`` so downstream scripts can
    quote the test number that corresponds to the val-selected config.
  * the run directory ``run_dir`` and ``prefix`` so collect_loss_matrix.py
    can skip its own HP search.

Usage::

    python experiments/collect_bench_p2_val.py
    python experiments/collect_bench_p2_val.py --problem knapsack
    python experiments/collect_bench_p2_val.py --method perturb
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

# Regex for "val MSE (no solver)" lines emitted by --skip_solver_eval runs.
_VAL_MSE_LINE = re.compile(
    r"Iter\s+(\d+),\s*val MSE \(no solver\):\s*([0-9eE+\-\.]+)"
)

# ---- Configuration (mirrors collect_loss_matrix.py) ----

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "pg_misspec", "shortestpath"]

ALL_METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
               "qptl", "cpLayer", "dad"]

PROB_ARG = {
    "knapsack": "knapsack", "knapsack-real": "knapsack",
    "energy": "energy", "budgetalloc": "budgetalloc",
    "cubic": "cubic", "bipartitematching": "bipartitematching",
    "portfolio": "portfolio", "asurv": "asurv", "cook_county": "cook_county",
    "speed_humps": "speed_humps", "sp_synth": "sp_synth",
    "sp_planted": "sp_planted", "pg_misspec": "pg_misspec",
    "shortestpath": "shortestpath",
}

PROB_VERSION = {
    "knapsack": "gen", "knapsack-real": "energy", "energy": "energy",
    "budgetalloc": "real", "cubic": "gen", "bipartitematching": "cora",
    "portfolio": "real", "asurv": "real", "cook_county": "real",
    "speed_humps": "real", "sp_synth": "synth", "sp_planted": "planted",
    "pg_misspec": "v3", "shortestpath": "warcraft",
}

METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      {"knapsack", "knapsack-real", "energy", "budgetalloc", "cubic",
                "bipartitematching", "portfolio", "asurv", "cook_county",
                "speed_humps", "sp_synth", "sp_planted", "pg_misspec"},
}

# Phase-2 sweeps: per method, a list of sub-sweeps. Each sub-sweep enumerates
# values for one HP; the prefix tag includes the *other* perturb HP at its
# default. The default HP value for each sweep (marked ``*`` in
# submit_bench_p2.sh) is what defines the "fixed" cell of the tuning-benefit
# table downstream; record it too.
HP_SWEEPS = {
    "dfl":      [dict(hp="dflalpha",    default="0.1",
                      vals=["0.001", "0.01", "0.1", "1.0", "10.0"],
                      tag_fn=lambda v: f"alpha{v}")],
    "blackbox": [dict(hp="lambd",       default="0.1",
                      vals=["0.01", "0.05", "0.1", "0.5", "1.0"],
                      tag_fn=lambda v: f"lam{v}")],
    "qptl":     [dict(hp="tau",         default="1.0",
                      vals=["0.1", "0.5", "1.0", "5.0", "10.0"],
                      tag_fn=lambda v: f"tau{v}")],
    "listLTR":  [dict(hp="tau",         default="1",
                      vals=["0.1", "0.5", "1", "5", "10"],
                      tag_fn=lambda v: f"tau{v}")],
    "lodl":     [dict(hp="num_samples", default="500",
                      vals=["100", "250", "500", "1000", "2000"],
                      tag_fn=lambda v: f"ns{v}")],
    "pg":       [dict(hp="sigma",       default="0.1",
                      vals=["0.01", "0.05", "0.1", "0.5", "1.0"],
                      tag_fn=lambda v: f"s{v.replace('.', 'p')}")],
    "perturb":  [dict(hp="sigma",       default="1.0",
                      vals=["0.1", "0.5", "1.0", "2.0", "5.0"],
                      tag_fn=lambda v: f"s{v.replace('.', 'p')}_n10"),
                 dict(hp="n_samples",   default="10",
                      vals=["5", "10", "25", "50", "100"],
                      tag_fn=lambda v: f"s1p0_n{v}")],
    "dad":      [dict(hp="stein_weight", default="1.0",
                      vals=["0.1", "0.5", "1.0", "2.0", "5.0"],
                      tag_fn=lambda v: f"sw{v.replace('.', 'p')}")],
}

USE_ABSOLUTE = {"portfolio"}
RESULTS_ROOT = "saved_records"
P1_BEST_PATH = "bench_p1_best_val.json"
OUT_PATH = "bench_p2_best_val.json"

# Multi-seed expansion (2026-05-21) — must match collect_bench_p1.py.
SEEDS = list(range(10))
SEED_EXCLUDED_METHODS = {"dad"}
SEED_EXCLUDED_PROBLEMS = {"energy", "shortestpath"}
SEED_EXCLUDED_PAIRS = {("perturb", "budgetalloc")}


def seeds_for(prob, method):
    if method in SEED_EXCLUDED_METHODS: return [0]
    if prob in SEED_EXCLUDED_PROBLEMS: return [0]
    if (method, prob) in SEED_EXCLUDED_PAIRS: return [0]
    return list(SEEDS)


def add_seed_suffix(prefix, seed):
    return prefix if seed == 0 else f"{prefix}_s{seed}"


def run_dir(prob, method, prefix):
    return os.path.join(RESULTS_ROOT, f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}",
                        method, prefix)


def p1_prefix(method, lr, batch):
    return f"bench_p1_{method}_{batch}_lr{lr}"


def p2_prefix(method, hp_tag, lr, batch):
    return f"bench_p2_{method}_{hp_tag}_{batch}_lr{lr}"


_TRAIN_MSE_LINE = re.compile(
    r"Iter\s+(\d+),\s*train MSE \(no solver\):\s*([0-9eE+\-\.]+)"
)


def _min_from_csv_eval_col(path):
    if not os.path.exists(path):
        return None
    try:
        min_v = None
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "eval" not in (reader.fieldnames or []):
                return None
            for row in reader:
                try:
                    v = float(row["eval"])
                except (TypeError, ValueError):
                    continue
                if min_v is None or v < min_v:
                    min_v = v
        return min_v
    except Exception:
        return None


def _min_from_log(path, regex):
    if not os.path.exists(path):
        return None
    try:
        min_v = None
        with open(path) as f:
            for line in f:
                m = regex.search(line)
                if not m:
                    continue
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                if min_v is None or v < min_v:
                    min_v = v
        return min_v
    except Exception:
        return None


def load_selection_signal(dirpath, method):
    """Load the selection signal from one run directory using the right reader
    for the method:
      mse_train → train MSE (CSV eval col, fallback log.txt)
      mse_val   → val MSE   (log.txt only — solver_valfreq=0)
      else      → val regret (val_logs.csv eval col, fallback val pred MSE log)
    Returns None if the run did not produce a usable signal.
    """
    if method == "mse_train":
        v = _min_from_csv_eval_col(os.path.join(dirpath, "train_logs.csv"))
        if v is None:
            v = _min_from_log(os.path.join(dirpath, "log.txt"), _TRAIN_MSE_LINE)
        return v
    if method == "mse_val":
        return _min_from_log(os.path.join(dirpath, "log.txt"), _VAL_MSE_LINE)
    # Default: val regret. Fallback to "val MSE (no solver)" for runs with
    # --skip_solver_eval (e.g. energy mse/dfl/identity).
    v = _min_from_csv_eval_col(os.path.join(dirpath, "val_logs.csv"))
    if v is None:
        v = _min_from_log(os.path.join(dirpath, "log.txt"), _VAL_MSE_LINE)
    return v


def load_min_val_regret(dirpath):
    """Backward-compat alias — defaults to val-regret reader."""
    return load_selection_signal(dirpath, method="")


def aggregate_seeds(prob, method, prefix_base):
    """Across all expected seeds for (prob, method), read each seed's selection
    signal from <prefix_base>[_s{N}] and aggregate.

    Returns dict with mean/std/n_done/n_total/seed_values/per_seed_run_dir, or
    None if no seed completed.
    """
    expected = seeds_for(prob, method)
    per_seed_vals = {}
    per_seed_dirs = {}
    for seed in expected:
        prefix = add_seed_suffix(prefix_base, seed)
        d = run_dir(prob, method, prefix)
        v = load_selection_signal(d, method)
        per_seed_dirs[seed] = d
        if v is not None:
            per_seed_vals[seed] = v
    if not per_seed_vals:
        return None
    arr = np.array(list(per_seed_vals.values()), dtype=float)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_done":  len(per_seed_vals),
        "n_total": len(expected),
        "seed_values":     {int(s): float(v) for s, v in per_seed_vals.items()},
        "per_seed_run_dir": {int(s): per_seed_dirs[s] for s in per_seed_dirs},
    }


def aggregate_test_regret(prob, method, prefix_base):
    """Mean test regret across all completed seeds at <prefix_base>.

    Returns dict with mean/std/seed_values, or None if no seed has results.npy.
    """
    expected = seeds_for(prob, method)
    per_seed = {}
    for seed in expected:
        prefix = add_seed_suffix(prefix_base, seed)
        d = run_dir(prob, method, prefix)
        v = load_test_regret(d, prob)
        if v is not None:
            per_seed[seed] = v
    if not per_seed:
        return None
    arr = np.array(list(per_seed.values()), dtype=float)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_done":    len(per_seed),
        "n_total":   len(expected),
        "seed_values": {int(s): float(v) for s, v in per_seed.items()},
    }


def load_test_regret(dirpath, prob):
    """Return scalar test regret (absolute for portfolio, else relative)."""
    rpath = os.path.join(dirpath, "results.npy")
    if not os.path.exists(rpath):
        return None
    try:
        r = np.load(rpath, allow_pickle=True)
        regret = np.asarray(r[1], dtype=float)
        opt    = np.asarray(r[0], dtype=float)
        abs_r  = float(np.mean(regret))
        if prob in USE_ABSOLUTE:
            return abs_r
        mean_opt = float(np.mean(np.abs(opt)))
        return abs_r / mean_opt if mean_opt > 0 else None
    except Exception:
        return None


def pick_winner(prob, method, lr, batch, p1_val=None):
    """
    For a Phase-2 method, enumerate every (sweep, value) candidate, aggregate
    its selection signal across all expected seeds, and pick the candidate with
    the lowest MEAN. For a non-Phase-2 method, aggregate the Phase-1 cell
    across seeds instead. Returns None on failure.

    Returned dict carries val=mean, std, seed_values, n_done/n_total (seeds at
    winner), and `run_dir` = seed-0 directory (for backward-compat with
    downstream consumers that don't know about seeds).
    """
    if method not in HP_SWEEPS:
        # PtO methods: no Phase-2 HP sweep; aggregate Phase-1 cell across seeds.
        prefix = p1_prefix(method, lr, batch)
        agg = aggregate_seeds(prob, method, prefix)
        if agg is None:
            return None
        # n_total here is n_total seeds (was 1 before); n_done likewise.
        return dict(val=agg["mean"], std=agg["std"], seed_values=agg["seed_values"],
                    hp_name=None, hp_value=None, hp_tag=None,
                    prefix=prefix, run_dir=run_dir(prob, method, prefix),
                    per_seed_run_dir=agg["per_seed_run_dir"],
                    n_done=agg["n_done"], n_total=agg["n_total"])

    candidates = []
    expected_seeds = len(seeds_for(prob, method))
    n_hp_total = 0
    for sweep in HP_SWEEPS[method]:
        for v in sweep["vals"]:
            n_hp_total += 1
            tag = sweep["tag_fn"](v)
            prefix = p2_prefix(method, tag, lr, batch)
            agg = aggregate_seeds(prob, method, prefix)
            if agg is None:
                continue
            candidates.append((agg["mean"], sweep["hp"], v, tag, prefix, agg))

    if not candidates:
        # No Phase-2 run exists at this (lr, batch). Happens when Phase-1's
        # winning arm was never the one the Phase-2 sweep was launched against
        # (e.g. listLTR x shortestpath, whose default-batch arm is withdrawn by
        # collect_bench_p1.BATCH_EXCLUDED_PAIRS). Fall back to the Phase-1 cell
        # -- method HP at its config default -- rather than dropping the cell
        # from the table entirely.
        prefix = p1_prefix(method, lr, batch)
        agg = aggregate_seeds(prob, method, prefix)
        if agg is None:
            return None
        return dict(val=agg["mean"], std=agg["std"], seed_values=agg["seed_values"],
                    hp_name=None, hp_value=None, hp_tag=None,
                    prefix=prefix, run_dir=run_dir(prob, method, prefix),
                    per_seed_run_dir=agg["per_seed_run_dir"],
                    n_done=agg["n_done"], n_total=agg["n_total"],
                    n_hp_done=0, n_hp_total=n_hp_total,
                    expected_seeds=expected_seeds, p2_fallback_to_p1=True)

    candidates.sort(key=lambda x: x[0])
    mean, hp_name, hp_value, tag, prefix, agg = candidates[0]
    return dict(val=mean, std=agg["std"], seed_values=agg["seed_values"],
                hp_name=hp_name, hp_value=hp_value, hp_tag=tag,
                prefix=prefix, run_dir=run_dir(prob, method, prefix),
                per_seed_run_dir=agg["per_seed_run_dir"],
                n_done=agg["n_done"], n_total=agg["n_total"],
                n_hp_done=len(candidates), n_hp_total=n_hp_total,
                expected_seeds=expected_seeds)


def fixed_default_entry(prob, method, lr, batch):
    """Default-HP cell (used by tuning-benefit table). Aggregated over seeds."""
    if method not in HP_SWEEPS:
        prefix = p1_prefix(method, lr, batch)
        agg = aggregate_seeds(prob, method, prefix)
        if agg is None:
            return None
        return dict(val=agg["mean"], std=agg["std"], seed_values=agg["seed_values"],
                    hp_name=None, hp_value=None, hp_tag=None,
                    prefix=prefix, run_dir=run_dir(prob, method, prefix),
                    n_done=agg["n_done"], n_total=agg["n_total"])

    for sweep in HP_SWEEPS[method]:
        tag = sweep["tag_fn"](sweep["default"])
        prefix = p2_prefix(method, tag, lr, batch)
        agg = aggregate_seeds(prob, method, prefix)
        if agg is not None:
            return dict(val=agg["mean"], std=agg["std"], seed_values=agg["seed_values"],
                        hp_name=sweep["hp"], hp_value=sweep["default"], hp_tag=tag,
                        prefix=prefix, run_dir=run_dir(prob, method, prefix),
                        n_done=agg["n_done"], n_total=agg["n_total"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--p1_best_json", default=P1_BEST_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.p1_best_json):
        print(f"error: {args.p1_best_json} not found. "
              f"Run `python experiments/collect_bench_p1.py --metric val` first.")
        sys.exit(1)

    with open(args.p1_best_json) as f:
        p1_best = json.load(f)

    problems = [args.problem] if args.problem else PROBLEMS
    methods  = [args.method]  if args.method  else ALL_METHODS

    out = {}
    for method in methods:
        out[method] = {}
        for prob in problems:
            allowed = METHOD_PROBLEMS.get(method)
            if allowed is not None and prob not in allowed:
                continue
            cfg = p1_best.get(method, {}).get(prob)
            if not cfg:
                continue
            lr = cfg.get("lr")
            batch = cfg.get("batch")
            if lr is None or batch is None:
                continue

            winner = pick_winner(prob, method, lr, batch, p1_val=cfg.get("val"))
            if winner is None:
                continue
            test_agg = aggregate_test_regret(prob, method, winner["prefix"])
            entry = {
                "lr": lr, "batch": batch,
                "hp_name":  winner["hp_name"],
                "hp_value": winner["hp_value"],
                "hp_tag":   winner["hp_tag"],
                "prefix":   winner["prefix"],
                "run_dir":  winner["run_dir"],
                # val is the multi-seed MEAN of the selection signal.
                "val":  round(winner["val"], 6),
                "val_std":     round(winner["std"], 6),
                "n_done":      winner["n_done"],   # seeds at winning HP
                "n_total":     winner["n_total"],  # seeds expected at winning HP
                "seed_values": winner["seed_values"],
                "phase":   1 if (method not in HP_SWEEPS
                                 or winner.get("p2_fallback_to_p1")) else 2,
            }
            if winner.get("p2_fallback_to_p1"):
                # Phase-2 sweep has no runs at this (lr, batch); HP is the
                # method default from the Phase-1 run.
                entry["p2_fallback_to_p1"] = True
            if "n_hp_done" in winner:
                entry["n_hp_done"]  = winner["n_hp_done"]
                entry["n_hp_total"] = winner["n_hp_total"]
            if test_agg is not None:
                entry["test"]               = round(test_agg["mean"], 6)
                entry["test_std"]           = round(test_agg["std"], 6)
                entry["test_seed_values"]   = test_agg["seed_values"]
                entry["test_n_done"]        = test_agg["n_done"]
            else:
                entry["test"] = None
            # Also record the fixed-default cell so the tuning-benefit script
            # can read it without re-doing path math.
            if method in HP_SWEEPS:
                fixed = fixed_default_entry(prob, method, lr, batch)
                if fixed is not None:
                    fixed_test = aggregate_test_regret(prob, method, fixed["prefix"])
                    entry["fixed"] = {
                        "hp_name":  fixed["hp_name"],
                        "hp_value": fixed["hp_value"],
                        "hp_tag":   fixed["hp_tag"],
                        "prefix":   fixed["prefix"],
                        "run_dir":  fixed["run_dir"],
                        "val":      round(fixed["val"], 6),
                        "val_std":  round(fixed["std"], 6),
                        "seed_values": fixed["seed_values"],
                        "n_done":   fixed["n_done"],
                        "n_total":  fixed["n_total"],
                        "test":     None if fixed_test is None else round(fixed_test["mean"], 6),
                        "test_std": None if fixed_test is None else round(fixed_test["std"], 6),
                    }
            out[method][prob] = entry

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    n_full = n_partial = n_missing = 0
    for method in ALL_METHODS:
        allowed = METHOD_PROBLEMS.get(method)
        for prob in PROBLEMS:
            if allowed is not None and prob not in allowed:
                continue
            entry = out.get(method, {}).get(prob)
            if entry is None:
                n_missing += 1
            elif entry["n_done"] == entry["n_total"]:
                n_full += 1
            else:
                n_partial += 1
    print(f"wrote {args.out}")
    print(f"full: {n_full}   partial: {n_partial}   missing: {n_missing}")


if __name__ == "__main__":
    main()
