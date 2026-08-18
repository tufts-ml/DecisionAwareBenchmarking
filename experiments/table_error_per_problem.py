"""
table_error_per_problem.py
--------------------------
Readable per-problem version of the train / val / test error tables.

Where ``table_error_grid.py`` packs 15 methods x 13 problems into one wide
landscape grid (3 numbers per cell), this script emits one compact table
per problem (13 in total) with 15 method rows and 6 columns:

    Method | Pred Train | Pred Val | Pred Test | Reg Train | Reg Val | Reg Test

All 13 sub-tables go into a single ``results/tables/error_per_problem.tex`` file
(``\\input`` once from the thesis chapter). Each sub-table has its own
caption and ``\\label{tab:err-<prob>}``. The lowest test value in each of
the two metric blocks is bolded so the chapter's "best method per task" is
visible at a glance.

Source: ``loss_matrix.json`` (rebuilt from val-selected configs by
``experiments/collect_loss_matrix.py --p2_best_json bench_p2_best_val.json``).
"""

import argparse
import json
import os
import sys

import numpy as np

LOSS_MATRIX_PATH = "loss_matrix.json"
DEFAULT_OUT = "results/tables/error_per_problem.tex"

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "pg_misspec", "shortestpath"]

# Display name without trailing period; the caption template adds one.
PROBLEM_DISPLAY = {
    "knapsack": "Knapsack (synth)",
    "knapsack-real": "Knapsack (energy)",
    "energy": "Energy scheduling",
    "budgetalloc": "Budget allocation",
    "cubic": "Cubic Top-$K$",
    "bipartitematching": "Bipartite matching",
    "portfolio": "Portfolio optimisation",
    "asurv": "Aerial survey",
    "cook_county": "Cook County opioid",
    "speed_humps": "NYC speed humps",
    "sp_synth": "Shortest path (synth)",
    "sp_planted": "Shortest path (planted)",
    "pg_misspec": "PG mis-specification",
    "shortestpath": "Warcraft shortest path",
}

ALL_METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg"]

METHOD_LABELS = {
    "mse": "MSE",
    "mse_train": r"MSE\textsubscript{train}",
    "mse_val":   r"MSE\textsubscript{val}",
    "dfl": "DFL", "identity": "Identity",
    "spo": r"SPO\textsuperscript{+}", "nce": "NCE", "blackbox": "Blackbox",
    "pointLTR": "ptLTR", "pairLTR": "prLTR", "listLTR": "lsLTR",
    "lodl": "LODL", "perturb": "DPO", "pg": "PG",
}

PRED_KEYS = ("train_pred_loss", "val_pred_loss", "test_pred_loss")
REG_KEYS  = ("train_regret",     "val_regret",    "test_regret_abs")


def _scale_for_column(values):
    """Return (digits, multiplier_exponent) suited to a column's magnitudes.
    multiplier_exponent is the power of 10 to multiply values by for display
    (column header gets a "(x 10^k)" suffix when nonzero).

    Picks the format so that even the smallest (nonzero) value in the column
    stays readable -- avoids "0.000" underflow when a column mixes small and
    large values (e.g. MSE 3e-4 alongside Identity 1.04 on budgetalloc)."""
    finite = [abs(v) for v in values
              if v is not None and isinstance(v, (int, float))
              and np.isfinite(v) and v != 0]
    if not finite:
        return 3, 0
    mag = np.median(finite)
    smallest = min(finite)

    if mag >= 1000:
        digits, exp = 0, 0       # "583"
    elif mag >= 100:
        digits, exp = 1, 0       # "583.2"
    elif mag >= 10:
        digits, exp = 2, 0       # "58.32"
    elif mag >= 1:
        digits, exp = 3, 0       # "5.832"
    elif mag >= 0.01:
        digits, exp = 4, 0       # "0.0584"
    elif mag >= 1e-4:
        digits, exp = 2, 3       # x10^3, header "(x 10^-3)"
    else:
        digits, exp = 2, 6

    # Bump precision if the smallest value would still underflow: need at
    # least one significant digit for it after scaling.
    scaled_smallest = smallest * (10 ** exp)
    if scaled_smallest > 0 and digits > 0:
        needed = max(0, -int(np.floor(np.log10(scaled_smallest))))
        if needed > digits:
            digits = min(needed + 1, 6)
    return digits, exp


def _fmt_value(v, digits):
    if v is None:
        return "--"
    if isinstance(v, str):
        return r"$\infty$" if v == "inf" else v
    if not np.isfinite(v):
        return "--"
    if v == 0 and digits > 0:
        return "0"
    if abs(v) >= 1e5:
        mantissa, exp = f"{v:.2e}".split("e")
        return rf"${mantissa}{{\times}}10^{{{int(exp)}}}$"
    return f"{v:.{digits}f}"


def _gather_column(loss_matrix, prob, key, methods):
    """Return list of (method, raw_value) for one column of one problem."""
    out = []
    for m in methods:
        entry = loss_matrix.get(prob, {}).get(m)
        if entry is None:
            out.append((m, None))
            continue
        v = entry.get("metrics", {}).get(key)
        out.append((m, v))
    return out


def _argmin_method(col):
    """Return the method whose value is the minimum (None-safe)."""
    finite = [(m, v) for m, v in col
              if v is not None and isinstance(v, (int, float))
              and np.isfinite(v)]
    if not finite:
        return None
    return min(finite, key=lambda x: x[1])[0]


def _scale_header(label, exp):
    if exp == 0:
        return label
    return rf"{label} ($\times 10^{{-{exp}}}$)"


def render_one_problem(loss_matrix, prob, methods=None):
    if methods is None:
        methods = ALL_METHODS
    # Per-column display scale, computed from raw values across the FULL
    # method list -- --exclude drops rows and bolding candidates but must not
    # change how the remaining rows are formatted.
    pred_cols = [_gather_column(loss_matrix, prob, k, ALL_METHODS) for k in PRED_KEYS]
    reg_cols  = [_gather_column(loss_matrix, prob, k, ALL_METHODS) for k in REG_KEYS]

    pred_scales = [_scale_for_column([v for _, v in col]) for col in pred_cols]
    reg_scales  = [_scale_for_column([v for _, v in col]) for col in reg_cols]

    # Identify the best (min) test value in each metric for bolding, among
    # the displayed methods only.
    shown = set(methods)
    pred_best = _argmin_method([mv for mv in pred_cols[2] if mv[0] in shown])
    reg_best  = _argmin_method([mv for mv in reg_cols[2] if mv[0] in shown])

    rows = []
    for m in methods:
        cells = [METHOD_LABELS[m]]
        for col, (digits, exp) in zip(pred_cols, pred_scales):
            v = dict(col)[m]
            scaled = (v * (10 ** exp)) if (v is not None and isinstance(v, (int, float)) and np.isfinite(v)) else v
            cells.append(_fmt_value(scaled, digits))
        for col, (digits, exp) in zip(reg_cols, reg_scales):
            v = dict(col)[m]
            scaled = (v * (10 ** exp)) if (v is not None and isinstance(v, (int, float)) and np.isfinite(v)) else v
            cells.append(_fmt_value(scaled, digits))

        # Bold the test cells when this method is the per-metric winner.
        # Columns: 0=method, 1-3=pred (train, val, test), 4-6=reg (train, val, test)
        if m == pred_best:
            cells[3] = rf"\textbf{{{cells[3]}}}"
        if m == reg_best:
            cells[6] = rf"\textbf{{{cells[6]}}}"
        rows.append(" & ".join(cells) + r" \\")

    # Sub-headers respect the scale annotations.
    pred_train_h = _scale_header("Train", pred_scales[0][1])
    pred_val_h   = _scale_header("Val",   pred_scales[1][1])
    pred_test_h  = _scale_header("Test",  pred_scales[2][1])
    reg_train_h  = _scale_header("Train", reg_scales[0][1])
    reg_val_h    = _scale_header("Val",   reg_scales[1][1])
    reg_test_h   = _scale_header("Test",  reg_scales[2][1])

    head = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{Train / val / test error for {PROBLEM_DISPLAY[prob]}. "
        r"Bold = best test value per metric. Hyperparameters per row: "
        r"val-best (LR, batch, method-specific HP). "
        r"Prediction loss is two-stage MSE/BCE. Decision regret is mean "
        r"absolute regret (lower is better).}" "\n"
        rf"\label{{tab:err-{prob.replace('_','-')}}}" "\n"
        r"\setlength{\tabcolsep}{4pt}" "\n"
        r"\renewcommand{\arraystretch}{1.05}" "\n"
        r"\begin{tabular}{@{}lrrrrrr@{}}" "\n"
        r"\toprule" "\n"
        r"& \multicolumn{3}{c}{Prediction loss} & \multicolumn{3}{c}{Decision regret} \\" "\n"
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
        rf"Method & {pred_train_h} & {pred_val_h} & {pred_test_h} & "
        rf"{reg_train_h} & {reg_val_h} & {reg_test_h} \\" "\n"
        r"\midrule"
    )
    body = "\n".join(rows)
    foot = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )
    return f"{head}\n{body}\n{foot}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_matrix", default=LOSS_MATRIX_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--problem", default=None,
                    help="Restrict to one problem (otherwise all 13).")
    ap.add_argument("--problems", nargs="*", default=None, metavar="PROB",
                    help="Restrict to a subset of problems; emitted in "
                         "loss_matrix order regardless of the order given "
                         "(default: all).")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="METHOD",
                    help="Method keys to drop from every sub-table. Applied "
                         "before best-value bolding (excluded methods cannot "
                         "be bolded as best); column display scales are still "
                         "computed over the full method list, so the "
                         "remaining rows keep their formatting. Keys not in "
                         "the method list are ignored.")
    args = ap.parse_args()

    if not os.path.exists(args.loss_matrix):
        print(f"error: {args.loss_matrix} not found. Run "
              f"`python experiments/collect_loss_matrix.py "
              f"--p2_best_json bench_p2_best_val.json` first.")
        sys.exit(1)
    with open(args.loss_matrix) as f:
        loss_matrix = json.load(f)

    if args.problem:
        problems = [args.problem]
    elif args.problems is not None:
        unknown = [p for p in args.problems if p not in PROBLEMS]
        if unknown:
            print(f"error: unknown problem(s): {' '.join(unknown)}")
            sys.exit(1)
        problems = [p for p in PROBLEMS if p in args.problems]
    else:
        problems = PROBLEMS

    methods = [m for m in ALL_METHODS if m not in set(args.exclude)]
    chunks = [render_one_problem(loss_matrix, p, methods) for p in problems]
    tex = "\n".join(chunks)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"wrote {args.out}  ({len(chunks)} sub-tables, "
          f"{len(methods)} methods x 6 cols each)")


if __name__ == "__main__":
    main()
