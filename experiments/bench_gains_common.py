"""
bench_gains_common.py
---------------------
Shared grid-builder + selection logic for the tuning-gain analyses
(consumed by ``table_rank_counterfactual.py``): per method x task it builds
the full sweep grid over both batch arms x 5 LRs plus the Phase-2 method-HP
cells, in three seed-aggregation modes.

It reads the full sweep directly from ``saved_records/`` by reusing the path
and IO helpers in ``collect_bench_p1.py`` / ``collect_bench_p2_val.py`` -- the
published ``bench_p1_best_val.json`` / ``bench_p2_best_val.json`` only carry the
winning cell, but these tables need *both* batch arms and the default-HP cell.

Selection principle (no test leakage): hyperparameters (LR, batch, method-HP)
are always chosen by the validation signal; we then report the *test* relative
regret of that val-selected config.

Batch arms (see ``submit_bench_p1.sh``):
  * ``default`` = the original benchmark regime: full-batch ``gd`` for most
    methods, single-instance ``bs=1`` sgd for SPO/NCE/LTR.
  * ``alt``     = minibatch ``sgd bs=32`` (bs=4 for bipartitematching).
So "the gain from minibatching" is exactly ``default -> alt``.

Three seed-aggregation modes:
  * ``seed0`` -- use model-init seed 0 only.
  * ``best``  -- per (method, problem) pick the single seed whose default-batch,
    LR-tuned **validation** signal is lowest; report that fixed seed everywhere
    (the most charitable single run, chosen without touching test).
  * ``mean``  -- mean across all completed seeds (reproduces the JSONs).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_bench_p1 as p1mod
import collect_bench_p2_val as p2mod

# ---- Re-exported configuration ----------------------------------------------

PROBLEMS = p1mod.PROBLEMS
LRS = p1mod.LRS
BATCH_LABELS = p1mod.BATCH_LABELS                      # ["default", "alt"]
METHOD_PROBLEMS = p1mod.METHOD_PROBLEMS
USE_ABSOLUTE = p1mod.USE_ABSOLUTE                      # {"portfolio"}
HP_SWEEPS = p2mod.HP_SWEEPS

# All methods we report on (order = table row order). Mirrors collect order but
# drops nothing; qptl/cpLayer are LP-only and handled by METHOD_PROBLEMS.
ALL_METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce",
               "blackbox", "pointLTR", "pairLTR", "listLTR", "lodl", "perturb",
               "pg", "qptl", "cpLayer", "dad"]

# Methods with a Phase-2 method-specific HP sweep (-> can have a P2 gain).
P2_TUNABLE = set(HP_SWEEPS.keys())   # dfl blackbox qptl listLTR lodl pg perturb dad

SEED_MODES = ["seed0", "best", "mean"]

METHOD_LABELS = {
    "mse": "MSE", "mse_train": r"MSE\textsubscript{tr}",
    "mse_val": r"MSE\textsubscript{val}",
    "dfl": "DFL", "identity": "Identity",
    "spo": r"SPO\textsuperscript{+}", "nce": "NCE", "blackbox": "Blackbox",
    "pointLTR": "ptLTR", "pairLTR": "prLTR", "listLTR": "lsLTR",
    "lodl": "LODL", "perturb": "DPO", "pg": "PG", "qptl": "QPTL",
    "cpLayer": "cpLayer", "dad": "DAD",
}

PROB_LABELS = {
    "knapsack": "KS", "knapsack-real": "KS-E", "energy": "En",
    "budgetalloc": "BA", "cubic": "Cu", "bipartitematching": "BM",
    "portfolio": "Pf", "asurv": "AS", "cook_county": "CC",
    "speed_humps": "SH", "sp_synth": r"sp\textsubscript{s}",
    "sp_planted": r"sp\textsubscript{p}", "pg_misspec": r"pg\textsubscript{ms}",
    "shortestpath": "SP-W",
}

PROB_FULL = {
    "knapsack": "Knapsack (synth)", "knapsack-real": "Knapsack (energy)",
    "energy": "Energy scheduling", "budgetalloc": "Budget allocation",
    "cubic": "Cubic Top-$K$", "bipartitematching": "Bipartite matching",
    "portfolio": "Portfolio", "asurv": "asurv", "cook_county": "cook\\_county",
    "speed_humps": "speed\\_humps", "sp_synth": "Shortest path (synth)",
    "sp_planted": "Shortest path (planted)", "pg_misspec": "PG misspec",
    "shortestpath": "Warcraft shortest path",
}

# ---- Original Geng et al. benchmark numbers (Table 3) ------------------------
# Relative regret (%) for all problems except portfolio, which is reported as
# absolute regret (matches USE_ABSOLUTE). "-" in the paper -> entry absent.
GENG_PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
                 "cubic", "bipartitematching", "portfolio"]
GENG_METHODS = ["mse", "dfl", "blackbox", "identity", "cpLayer", "spo", "nce",
                "pointLTR", "pairLTR", "listLTR", "lodl"]

# GENG[method][problem] = published value (percent, except portfolio absolute).
GENG = {
    "mse":      {"knapsack": 6.595, "knapsack-real": 8.745, "energy": 1.793,
                 "budgetalloc": 20.332, "cubic": 0.110, "bipartitematching": 92.963,
                 "portfolio": 0.243},
    "dfl":      {"knapsack": 11.744, "knapsack-real": 8.353, "energy": 6.272,
                 "budgetalloc": 35.970, "cubic": 1.974, "bipartitematching": 91.364,
                 "portfolio": 0.380},
    "blackbox": {"knapsack": 24.274, "knapsack-real": 35.705, "energy": 6.503,
                 "budgetalloc": 26.905, "cubic": 13.944, "bipartitematching": 91.988,
                 "portfolio": 0.286},
    "identity": {"knapsack": 31.874, "knapsack-real": 17.156, "energy": 5.690,
                 "budgetalloc": 14.799, "cubic": 13.944, "bipartitematching": 91.868,
                 "portfolio": 0.280},
    "cpLayer":  {"knapsack": 24.769, "knapsack-real": 36.402,
                 "bipartitematching": 92.007, "portfolio": 0.309},
    "spo":      {"knapsack": 6.223, "knapsack-real": 8.407, "energy": 1.505,
                 "budgetalloc": 5.559, "cubic": 160.408, "bipartitematching": 93.327,
                 "portfolio": 0.245},
    "nce":      {"knapsack": 13.438, "knapsack-real": 11.932, "energy": 1.663,
                 "budgetalloc": 9.979, "cubic": 160.408, "bipartitematching": 92.622,
                 "portfolio": 0.367},
    "pointLTR": {"knapsack": 6.402, "knapsack-real": 8.236, "energy": 4.548,
                 "budgetalloc": 69.663, "cubic": 1.149, "bipartitematching": 91.035,
                 "portfolio": 0.214},
    "pairLTR":  {"knapsack": 7.820, "knapsack-real": 9.022, "energy": 1.540,
                 "budgetalloc": 5.958, "cubic": 5.072, "bipartitematching": 92.285,
                 "portfolio": 0.255},
    "listLTR":  {"knapsack": 6.031, "knapsack-real": 8.083, "energy": 1.551,
                 "budgetalloc": 5.742, "cubic": 0.193, "bipartitematching": 91.831,
                 "portfolio": 0.249},
    "lodl":     {"knapsack": 6.044, "knapsack-real": 9.567, "energy": 1.786,
                 "budgetalloc": 25.700, "cubic": 0.172, "bipartitematching": 91.113,
                 "portfolio": 0.160},
}


# ---- Grid construction ------------------------------------------------------

def build_p1_grid(prob, method):
    """Full Phase-1 grid for one (prob, method).

    Returns ``{(batch, lr): {seed: {"val": v, "test": t}}}`` over every cell that
    produced at least a val or a test number. ``val`` is the validation
    selection signal (val regret / val MSE / train MSE per method); ``test`` is
    the test relative regret (absolute regret for portfolio).
    """
    grid = {}
    for batch in BATCH_LABELS:
        for lr in LRS:
            cell = {}
            for s in p1mod.seeds_for(prob, method):
                val, _ = p1mod.load_signal(prob, method, batch, lr, s, "val")
                test, _ = p1mod.load_signal(prob, method, batch, lr, s, "test")
                if val is not None or test is not None:
                    cell[s] = {"val": val, "test": test}
            if cell:
                grid[(batch, lr)] = cell
    return grid


def _read_p2_cell(prob, method, tag, lr, batch):
    """Per-seed val+test for one Phase-2 HP candidate at (lr, batch)."""
    cell = {}
    for s in p2mod.seeds_for(prob, method):
        prefix = p2mod.add_seed_suffix(p2mod.p2_prefix(method, tag, lr, batch), s)
        d = p2mod.run_dir(prob, method, prefix)
        val = p2mod.load_selection_signal(d, method)
        test = p2mod.load_test_regret(d, prob)
        if val is not None or test is not None:
            cell[s] = {"val": val, "test": test}
    return cell


def build_p2_grid(prob, method, lr, batch):
    """Phase-2 HP grid at the val-best (lr, batch).

    Returns ``{"cells": {tag: {seed: {...}}}, "default_tag": tag}`` or None if the
    method has no Phase-2 sweep. ``default_tag`` marks the default-HP ("fixed")
    candidate. Tags from perturb's two sub-sweeps share the common default
    ``s1p0_n10`` and dedupe naturally.
    """
    if method not in HP_SWEEPS:
        return None
    cells = {}
    default_tag = None
    for sw in HP_SWEEPS[method]:
        dft = sw["tag_fn"](sw["default"])
        if default_tag is None:
            default_tag = dft
        for v in sw["vals"]:
            tag = sw["tag_fn"](v)
            if tag in cells:
                continue
            cell = _read_p2_cell(prob, method, tag, lr, batch)
            if cell:
                cells[tag] = cell
    if not cells:
        return None
    return {"cells": cells, "default_tag": default_tag}


# ---- Seed reduction & selection ---------------------------------------------

def chosen_seed(p1_grid):
    """Best-validating seed: argmin over seeds of (min over LR of default-batch
    val). Falls back to the alt batch if no default-batch cell exists (e.g.
    perturb on shortestpath). Returns None if the grid is empty.
    """
    for arm in ("default", "alt"):
        per_seed = {}
        for (batch, lr), cell in p1_grid.items():
            if batch != arm:
                continue
            for s, rec in cell.items():
                v = rec["val"]
                if v is None:
                    continue
                if s not in per_seed or v < per_seed[s]:
                    per_seed[s] = v
        if per_seed:
            return min(per_seed, key=lambda s: per_seed[s])
    return None


def reduce_cell(cell, mode, seed):
    """Collapse a ``{seed: {val, test}}`` cell to scalar (val, test) per mode."""
    if not cell:
        return None, None
    if mode == "seed0":
        rec = cell.get(0)
        return (None, None) if rec is None else (rec["val"], rec["test"])
    if mode == "best":
        rec = cell.get(seed)
        return (None, None) if rec is None else (rec["val"], rec["test"])
    # mean: val and test aggregated independently over their completed seeds.
    vals = [r["val"] for r in cell.values() if r["val"] is not None]
    tests = [r["test"] for r in cell.values() if r["test"] is not None]
    vmean = float(np.mean(vals)) if vals else None
    tmean = float(np.mean(tests)) if tests else None
    return vmean, tmean


def best_lr_in_batch(p1_grid, batch, mode, seed):
    """Val-select the LR within one batch arm. Returns (val, test, lr) or None."""
    best = None
    for lr in LRS:
        cell = p1_grid.get((batch, lr))
        if not cell:
            continue
        v, t = reduce_cell(cell, mode, seed)
        if v is None:
            continue
        if best is None or v < best[0]:
            best = (v, t, lr)
    return best


def p1_selection(prob, method, mode):
    """Returns dict with full-batch (``fb``), minibatch (``mb``) and overall
    val-best (``overall``) cells. Each is (val, test, lr) or None. ``overall``
    additionally carries its batch label as a 4th element.
    """
    grid = build_p1_grid(prob, method)
    seed = chosen_seed(grid) if mode == "best" else None
    fb = best_lr_in_batch(grid, "default", mode, seed)
    mb = best_lr_in_batch(grid, "alt", mode, seed)
    overall = None
    for cand, lab in ((fb, "default"), (mb, "alt")):
        if cand is None:
            continue
        if overall is None or cand[0] < overall[0]:
            overall = (cand[0], cand[1], cand[2], lab)
    return {"grid": grid, "seed": seed, "fb": fb, "mb": mb, "overall": overall}


def p2_selection(prob, method, mode, lr, batch):
    """Val-best tuned HP vs default ("fixed") HP at (lr, batch).

    Returns dict ``{"tuned": (val, test, tag), "fixed": (val, test)}`` or None.
    ``lr``/``batch`` should be the cell where Phase-2 was actually run (the
    val-best (lr, batch) from ``bench_p1_best_val.json``).
    """
    if method not in HP_SWEEPS:
        return None
    p2grid = build_p2_grid(prob, method, lr, batch)
    if p2grid is None:
        return None
    seed = chosen_seed(build_p1_grid(prob, method)) if mode == "best" else None
    tuned = None
    for tag, cell in p2grid["cells"].items():
        v, t = reduce_cell(cell, mode, seed)
        if v is None:
            continue
        if tuned is None or v < tuned[0]:
            tuned = (v, t, tag)
    fixed = None
    dft = p2grid["default_tag"]
    if dft in p2grid["cells"]:
        v, t = reduce_cell(p2grid["cells"][dft], mode, seed)
        if v is not None:
            fixed = (v, t)
    return {"tuned": tuned, "fixed": fixed}


# ---- Display helpers --------------------------------------------------------

def applies(method, prob):
    allowed = METHOD_PROBLEMS.get(method)
    return allowed is None or prob in allowed


def to_pct(test_val, prob):
    """Test relative regret -> percent for display; portfolio stays absolute."""
    if test_val is None:
        return None
    return test_val if prob in USE_ABSOLUTE else test_val * 100.0


def rel_gain(base, new):
    """Relative improvement (%) of ``new`` over ``base``: positive = better."""
    if base is None or new is None or base == 0:
        return None
    return (base - new) / abs(base) * 100.0


_P1_BEST_CACHE = {}


def p1_best_config(prob, method, path="bench_p1_best_val.json"):
    """(lr, batch) where Phase-2 was run for this cell, per bench_p1_best_val."""
    if path not in _P1_BEST_CACHE:
        with open(path) as f:
            _P1_BEST_CACHE[path] = json.load(f)
    cfg = _P1_BEST_CACHE[path].get(method, {}).get(prob)
    if not cfg:
        return None, None
    return cfg.get("lr"), cfg.get("batch")
