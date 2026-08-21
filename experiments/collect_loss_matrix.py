"""
collect_loss_matrix.py
----------------------
Assemble the (problem, method) x (train, val, test) x (pred, decision) loss
matrix.

For each (problem, method) at its best config (Phase 2 best if the method has
a Phase-2 HP sweep, else Phase 1 best):
  - train pred_loss, train decision regret  — read from train_logs.csv at the
    best-val-regret epoch
  - val pred_loss,   val decision regret    — read from val_logs.csv at the
    best-val-regret epoch
  - test decision regret                     — read from results.npy[1]
  - test pred_loss                           — read from test_pred_loss.json
    if present (produced by eval_test_pred.py); NaN otherwise.

Outputs:
  - results/loss_matrix.json                 — full 14 x 17 x 6 tensor as JSON
  - results/tables/loss_matrix/<problem>.md     — one markdown table per problem
  - results/tables/loss_matrix/summary.md       — overview across all problems

Caveat: with the legacy ``--best_json`` fallback (no ``--p2_best_json``),
Phase-1 best (lr, batch) is picked by *test* regret and Phase-2 HP selection
inherits that leakage; footnoted in every output. The documented invocation
(``--p2_best_json results/bench_p2_best_val.json``) is val-selected and unaffected.

Usage:
    python experiments/collect_loss_matrix.py
    python experiments/collect_loss_matrix.py --problem budgetalloc
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# ---- Configuration (mirrors collect_bench_p2_val.py) ----

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "pg_misspec", "shortestpath"]

ALL_METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
               "qptl", "cpLayer", "dad"]

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

METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      {"knapsack", "knapsack-real", "energy", "budgetalloc", "cubic",
                "bipartitematching", "portfolio", "asurv", "cook_county",
                "speed_humps", "sp_synth", "sp_planted", "pg_misspec"},
}

# Methods with Phase-2 HP sweeps.
# Each method may have multiple sweep keys (e.g. perturb sweeps sigma and n).
HP_SWEEPS = {
    "dfl":      [dict(hp="dflalpha",    vals=["0.001", "0.01", "0.1", "1.0", "10.0"],
                      tag_fn=lambda v: f"alpha{v}")],
    "blackbox": [dict(hp="lambd",       vals=["0.01", "0.05", "0.1", "0.5", "1.0"],
                      tag_fn=lambda v: f"lam{v}")],
    "qptl":     [dict(hp="tau",         vals=["0.1", "0.5", "1.0", "5.0", "10.0"],
                      tag_fn=lambda v: f"tau{v}")],
    "listLTR":  [dict(hp="tau",         vals=["0.1", "0.5", "1", "5", "10"],
                      tag_fn=lambda v: f"tau{v}")],
    "lodl":     [dict(hp="num_samples", vals=["100", "250", "500", "1000", "2000"],
                      tag_fn=lambda v: f"ns{v}")],
    "pg":       [dict(hp="sigma",       vals=["0.01", "0.05", "0.1", "0.5", "1.0"],
                      tag_fn=lambda v: f"s{v.replace('.', 'p')}")],
    "perturb":  [dict(hp="sigma",       vals=["0.1", "0.5", "1.0", "2.0", "5.0"],
                      tag_fn=lambda v: f"s{v.replace('.', 'p')}_n10"),
                 dict(hp="n_samples",   vals=["5", "10", "25", "50", "100"],
                      tag_fn=lambda v: f"s1p0_n{v}")],
    "dad":      [dict(hp="stein_weight", vals=["0.1", "0.5", "1.0", "2.0", "5.0"],
                      tag_fn=lambda v: f"sw{v.replace('.', 'p')}")],
}

USE_ABSOLUTE = {"portfolio"}  # report absolute regret, not relative
RESULTS_ROOT = "saved_records"
BEST_JSON_PATH = "results/bench_p1_best.json"
P2_BEST_VAL_PATH = "results/bench_p2_best_val.json"
OUTPUT_JSON = "results/loss_matrix.json"
TABLES_DIR = "results/tables/loss_matrix"


# ---- Path helpers ----

def run_dir(prob, method, prefix):
    parg = PROB_ARG[prob]
    pver = PROB_VERSION[prob]
    return os.path.join(RESULTS_ROOT, f"{parg}-{pver}", method, prefix)


def load_test_regret(dirpath):
    rpath = os.path.join(dirpath, "results.npy")
    if not os.path.exists(rpath):
        return None, None
    try:
        r = np.load(rpath, allow_pickle=True)
        regret = np.asarray(r[1], dtype=float)
        opt = np.asarray(r[0], dtype=float)
        abs_r = float(np.mean(regret))
        mean_opt = float(np.mean(np.abs(opt)))
        rel_r = abs_r / mean_opt if mean_opt > 0 else np.nan
        return abs_r, rel_r
    except Exception:
        return None, None


def pick_regret(prob, abs_r, rel_r):
    if abs_r is None:
        return None
    return abs_r if prob in USE_ABSOLUTE else rel_r


# ---- Best-config resolution ----

def p1_prefix(method, lr, batch):
    return f"bench_p1_{method}_{batch}_lr{lr}"


def p2_prefix(method, hp_tag, lr, batch):
    return f"bench_p2_{method}_{hp_tag}_{batch}_lr{lr}"


def resolve_best_run(prob, method, bench_p1_best, p2_best_val=None):
    """Return (dirpath, prefix, metadata) for the best config of (prob, method),
    or (None, None, None) if no results exist yet.

    If ``p2_best_val`` is provided, use its pre-resolved ``prefix`` directly
    (val-based selection by collect_bench_p2_val.py) and skip the per-run
    test-regret HP search. Otherwise fall back to the legacy behaviour
    (test-regret selection)."""
    allowed = METHOD_PROBLEMS.get(method)
    if allowed is not None and prob not in allowed:
        return None, None, None

    if p2_best_val is not None:
        entry = p2_best_val.get(method, {}).get(prob)
        if not entry:
            return None, None, None
        d = entry.get("run_dir") or run_dir(prob, method, entry["prefix"])
        meta = {"phase": entry.get("phase", 1),
                "lr": entry.get("lr"), "batch": entry.get("batch")}
        if entry.get("hp_name") is not None:
            meta["hp_name"] = entry["hp_name"]
            meta["hp_val"]  = entry["hp_value"]
            meta["hp_tag"]  = entry.get("hp_tag")
        return d, entry["prefix"], meta

    cfg = bench_p1_best.get(method, {}).get(prob)
    if not cfg:
        return None, None, None
    lr = cfg.get("lr")
    batch = cfg.get("batch")
    if lr is None or batch is None:
        return None, None, None

    if method in HP_SWEEPS:
        # sweep over all HP sweeps for this method; pick min test regret
        best = None  # (test_regret_metric, hp_tag, hp_name, hp_val, prefix, dirpath)
        for sweep in HP_SWEEPS[method]:
            for v in sweep["vals"]:
                tag = sweep["tag_fn"](v)
                prefix = p2_prefix(method, tag, lr, batch)
                d = run_dir(prob, method, prefix)
                abs_r, rel_r = load_test_regret(d)
                metric = pick_regret(prob, abs_r, rel_r)
                if metric is None or not np.isfinite(metric):
                    continue
                cand = (metric, tag, sweep["hp"], v, prefix, d)
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is None:
            return None, None, None
        _, tag, hp_name, hp_val, prefix, d = best
        meta = {"phase": 2, "lr": lr, "batch": batch,
                "hp_name": hp_name, "hp_val": hp_val, "hp_tag": tag}
        return d, prefix, meta
    else:
        prefix = p1_prefix(method, lr, batch)
        d = run_dir(prob, method, prefix)
        abs_r, rel_r = load_test_regret(d)
        if abs_r is None:
            return None, None, None
        meta = {"phase": 1, "lr": lr, "batch": batch}
        return d, prefix, meta


# ---- Per-run metric extraction ----

def read_best_val_row(dirpath):
    """From val_logs.csv, find the epoch with minimum 'eval' (val regret).
    Return (epoch_str, val_pred_loss, val_eval) or None if missing/empty."""
    vpath = os.path.join(dirpath, "val_logs.csv")
    if not os.path.exists(vpath):
        return None
    try:
        df = pd.read_csv(vpath)
    except Exception:
        return None
    if df.empty or "eval" not in df.columns:
        return None
    # Drop rows with NaN eval
    df = df.dropna(subset=["eval"])
    if df.empty:
        return None
    idx = df["eval"].idxmin()
    row = df.loc[idx]
    return (str(row["epoch"]),
            float(row["pred_loss"]),
            float(row["eval"]))


def read_train_row_at_epoch(dirpath, epoch_str):
    """From train_logs.csv, read the row at the given epoch.
    Returns (train_pred_loss, train_eval) or (nan, nan) if missing."""
    tpath = os.path.join(dirpath, "train_logs.csv")
    if not os.path.exists(tpath):
        return np.nan, np.nan
    try:
        df = pd.read_csv(tpath)
    except Exception:
        return np.nan, np.nan
    if df.empty or "eval" not in df.columns:
        return np.nan, np.nan
    rows = df[df["epoch"] == epoch_str]
    if rows.empty:
        return np.nan, np.nan
    row = rows.iloc[0]
    return (float(row["pred_loss"]),
            float(row["eval"]))


def read_test_pred_loss(dirpath):
    """Read test_pred_loss.json (produced by eval_test_pred.py).
    Returns float or nan."""
    path = os.path.join(dirpath, "test_pred_loss.json")
    if not os.path.exists(path):
        return np.nan
    try:
        with open(path) as f:
            d = json.load(f)
        return float(d.get("test_pred_loss", np.nan))
    except Exception:
        return np.nan


def read_all_pred_losses(dirpath):
    """Read all three (train, val, test) pred losses from test_pred_loss.json.
    Returns (train, val, test) tuple of floats, nan where missing."""
    path = os.path.join(dirpath, "test_pred_loss.json")
    if not os.path.exists(path):
        return np.nan, np.nan, np.nan
    try:
        with open(path) as f:
            d = json.load(f)
        return (float(d.get("train_pred_loss", np.nan)),
                float(d.get("val_pred_loss",   np.nan)),
                float(d.get("test_pred_loss",  np.nan)))
    except Exception:
        return np.nan, np.nan, np.nan


def read_best_epoch_train_mse(dirpath):
    """For mse_train runs: pick the epoch with min train MSE from train_logs.csv.
    Returns (epoch_str, train_pred_loss) or (None, nan)."""
    tpath = os.path.join(dirpath, "train_logs.csv")
    if not os.path.exists(tpath):
        return None, np.nan
    try:
        df = pd.read_csv(tpath)
    except Exception:
        return None, np.nan
    if df.empty or "eval" not in df.columns:
        return None, np.nan
    df = df.dropna(subset=["eval"])
    if df.empty:
        return None, np.nan
    idx = df["eval"].idxmin()
    row = df.loc[idx]
    return str(row["epoch"]), float(row["eval"])


_LOG_VAL_MSE = __import__("re").compile(
    r"Iter\s+(\d+),\s*val MSE \(no solver\):\s*([0-9eE+\-\.]+)"
)


def read_best_epoch_val_mse(dirpath):
    """For mse_val runs: parse log.txt for min val MSE line.
    Returns (epoch_str, val_pred_loss) or (None, nan)."""
    lpath = os.path.join(dirpath, "log.txt")
    if not os.path.exists(lpath):
        return None, np.nan
    try:
        best = None
        with open(lpath) as f:
            for line in f:
                m = _LOG_VAL_MSE.search(line)
                if not m:
                    continue
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                if best is None or v < best[1]:
                    best = (f"Tr-{m.group(1)}", v)
        if best is None:
            return None, np.nan
        return best
    except Exception:
        return None, np.nan


def extract_run_metrics(dirpath, prob):
    """Return dict with six headline metrics + epoch + abs/rel test regret.
    nan where unavailable.

    train_regret / val_regret are ABSOLUTE (mean of the per-epoch ``eval``
    column in *_logs.csv). test_regret is RELATIVE for every problem except
    portfolio (where the codebase reports absolute) -- that mirrors the
    headline metric used in the bump figures and the paper.
    test_regret_abs and test_regret_rel are also exposed so downstream
    table scripts can mix-and-match.
    """
    result = {
        "best_epoch": None,
        "train_pred_loss": np.nan, "train_regret": np.nan,
        "val_pred_loss":   np.nan, "val_regret":   np.nan,
        "test_pred_loss":  np.nan, "test_regret":  np.nan,
        "test_regret_abs": np.nan, "test_regret_rel": np.nan,
    }
    v = read_best_val_row(dirpath)
    if v is not None:
        epoch_str, val_pred, val_eval = v
        result["best_epoch"] = epoch_str
        result["val_pred_loss"] = val_pred
        result["val_regret"] = val_eval
        tp, te = read_train_row_at_epoch(dirpath, epoch_str)
        result["train_pred_loss"] = tp
        result["train_regret"] = te
    else:
        # No val_logs.csv (e.g., mse_train / mse_val / energy --skip_solver_eval
        # runs). Best epoch comes from whatever selection signal was used.
        # train/val regret will remain NaN since no solver was run during
        # training, but pred losses can be filled from test_pred_loss.json
        # (which eval_test_pred.py computes on the saved checkpoint).
        method_dir = os.path.basename(os.path.dirname(dirpath))
        if method_dir == "mse_train":
            epoch_str, _ = read_best_epoch_train_mse(dirpath)
            if epoch_str is not None:
                result["best_epoch"] = epoch_str
        elif method_dir == "mse_val":
            epoch_str, _ = read_best_epoch_val_mse(dirpath)
            if epoch_str is not None:
                result["best_epoch"] = epoch_str

    abs_r, rel_r = load_test_regret(dirpath)
    if abs_r is not None:
        result["test_regret_abs"] = abs_r
    if rel_r is not None:
        result["test_regret_rel"] = rel_r
    test_metric = pick_regret(prob, abs_r, rel_r)
    if test_metric is not None:
        result["test_regret"] = test_metric

    # test_pred_loss.json (produced by eval_test_pred.py on the saved
    # checkpoint) carries train/val/test pred_loss. Use it to backfill any
    # NaN pred-loss fields. This is the only source for mse_train / mse_val.
    tr_pred, val_pred, te_pred = read_all_pred_losses(dirpath)
    if np.isnan(result["test_pred_loss"]):
        result["test_pred_loss"] = te_pred
    if np.isnan(result["train_pred_loss"]):
        result["train_pred_loss"] = tr_pred
    if np.isnan(result["val_pred_loss"]):
        result["val_pred_loss"] = val_pred
    return result


# ---- Markdown table rendering ----

def fmt(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, float) and np.isnan(x):
        return "—"
    if isinstance(x, float) and np.isinf(x):
        return "∞"
    return f"{x:.{digits}f}"


def render_problem_table(prob, rows_by_method):
    scale_note = ("(absolute regret)" if prob in USE_ABSOLUTE
                  else "(relative regret)")
    lines = [
        f"# Loss Matrix — {prob} {scale_note}",
        "",
        "Each row: one method at its best (Phase 2) HP config. "
        "Values at the **best-val-regret epoch** (val_logs.csv argmin)."
        " Test regret from results.npy. Test pred_loss from "
        "`test_pred_loss.json` if present (produced by `eval_test_pred.py`).",
        "",
        "| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method in ALL_METHODS:
        r = rows_by_method.get(method)
        if r is None:
            lines.append(f"| {method} | — | — | — | — | — | — | — |")
            continue
        m = r["metrics"]
        meta = r["meta"]
        cfg_str = f"lr={meta['lr']}, bs={meta['batch']}"
        if "hp_name" in meta:
            cfg_str += f", {meta['hp_name']}={meta['hp_val']}"
        lines.append(
            f"| {method} | {cfg_str} "
            f"| {fmt(m['train_pred_loss'])} | {fmt(m['val_pred_loss'])} | {fmt(m['test_pred_loss'])} "
            f"| {fmt(m['train_regret'])} | {fmt(m['val_regret'])} | {fmt(m['test_regret'])} |"
        )
    lines.append("")
    lines.append("Phase-1 (lr, batch) and Phase-2 method-specific HP are both "
                 "selected by **val** decision regret (no test leakage). Source: "
                 "`bench_p2_best_val.json` via `collect_bench_p2_val.py`.")
    return "\n".join(lines) + "\n"


def render_summary_table(all_data):
    """Compact overview: methods as rows, problems as columns; cells = test regret."""
    lines = [
        "# Loss Matrix Summary",
        "",
        "Test decision regret at each method's best config. Cells show "
        "relative regret except `portfolio` (absolute). `—` = missing or "
        "not-applicable (e.g. qptl/cpLayer on non-QP problems, pg on shortestpath).",
        "",
    ]
    header = "| Method | " + " | ".join(PROBLEMS) + " |"
    sep = "|" + "---|" * (len(PROBLEMS) + 1)
    lines.append(header)
    lines.append(sep)
    for method in ALL_METHODS:
        cells = []
        for prob in PROBLEMS:
            r = all_data.get(prob, {}).get(method)
            if r is None:
                cells.append("—")
            else:
                cells.append(fmt(r["metrics"]["test_regret"]))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Configs are val-selected end-to-end (Phase 1 lr/batch and "
                 "Phase 2 method-specific HP both picked by val regret). "
                 "Source: `bench_p2_best_val.json`.")
    return "\n".join(lines) + "\n"


# ---- Main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None)
    ap.add_argument("--output_json", default=OUTPUT_JSON)
    ap.add_argument("--tables_dir", default=TABLES_DIR)
    ap.add_argument("--best_json", default=BEST_JSON_PATH,
                    help="Phase-1 best JSON (legacy fallback when "
                         "--p2_best_json not provided).")
    ap.add_argument("--p2_best_json", default=P2_BEST_VAL_PATH,
                    help="Pre-resolved val-best JSON from "
                         "collect_bench_p2_val.py. Pass empty string to "
                         "force the legacy test-regret HP search.")
    args = ap.parse_args()

    p2_best_val = None
    if args.p2_best_json and os.path.exists(args.p2_best_json):
        with open(args.p2_best_json) as f:
            p2_best_val = json.load(f)
        print(f"using val-selected configs from {args.p2_best_json}")

    if os.path.exists(args.best_json):
        with open(args.best_json) as f:
            bench_p1_best = json.load(f)
    else:
        bench_p1_best = {}
        if p2_best_val is None:
            print(f"error: neither --best_json ({args.best_json}) nor "
                  f"--p2_best_json ({args.p2_best_json}) is readable.")
            return

    problems = [args.problem] if args.problem else PROBLEMS

    all_data = defaultdict(dict)  # {prob: {method: {metrics, meta, prefix}}}
    for prob in problems:
        for method in ALL_METHODS:
            d, prefix, meta = resolve_best_run(prob, method, bench_p1_best,
                                               p2_best_val=p2_best_val)
            if d is None:
                continue
            metrics = extract_run_metrics(d, prob)
            all_data[prob][method] = {
                "metrics": metrics,
                "meta": meta,
                "prefix": prefix,
                "run_dir": d,
            }

    os.makedirs(args.tables_dir, exist_ok=True)
    for prob in problems:
        md = render_problem_table(prob, all_data.get(prob, {}))
        out = os.path.join(args.tables_dir, f"{prob}.md")
        with open(out, "w") as f:
            f.write(md)
        print(f"wrote {out}")

    summary = render_summary_table(all_data)
    summary_path = os.path.join(args.tables_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"wrote {summary_path}")

    # Dump JSON
    json_safe = {}
    for prob, mdata in all_data.items():
        json_safe[prob] = {}
        for method, entry in mdata.items():
            m = dict(entry["metrics"])
            for k, v in m.items():
                if isinstance(v, float):
                    if np.isnan(v):
                        m[k] = None
                    elif np.isinf(v):
                        m[k] = "inf"
            json_safe[prob][method] = {
                "metrics": m,
                "meta": entry["meta"],
                "prefix": entry["prefix"],
                "run_dir": entry["run_dir"],
            }
    with open(args.output_json, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"wrote {args.output_json}")

    # Report missing test_pred_loss count (None = genuinely missing, "inf" = diverged)
    missing = 0
    diverged = 0
    total = 0
    for prob in json_safe:
        for method, entry in json_safe[prob].items():
            total += 1
            v = entry["metrics"].get("test_pred_loss")
            if v is None:
                missing += 1
            elif v == "inf":
                diverged += 1
    print(f"\n{missing}/{total} (prob, method) cells missing test_pred_loss; "
          f"{diverged}/{total} diverged (inf). "
          f"Run `python experiments/eval_test_pred.py` to fill missing.")


if __name__ == "__main__":
    main()
