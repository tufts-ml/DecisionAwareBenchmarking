"""
table_best_hyperparams.py
-------------------------
Best (learning rate, batch regime, method-specific hyperparameter) per
method x task from the Phase-1/2 sweep, as selected by minimum 10-seed-mean
validation decision regret.

Reads bench_p2_best_val.json (primary; every method has an entry there, with
phase-1 fallbacks for PtO-only methods) and bench_p1_best_val.json (fallback
for any cell absent from the Phase-2 file). Batch-arm labels ("default"/"alt")
are resolved to their real meaning per method x task (full batch / bs=1 /
bs=32 / bs=4); the arm semantics mirror slurm/submit_bench_p1.sh.

Outputs (to --outdir, default results/tables/):
  best_hyperparams.csv   long format: method,task,lr,batch,hp_name,hp_value
  best_hyperparams.md    grid (methods x 14 tasks) with legend
  best_hyperparams.tex   booktabs table (two stacked 7-task sub-tables)

Usage:
    python experiments/table_best_hyperparams.py
    python experiments/table_best_hyperparams.py --outdir results/tables
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_bench_p1 as p1mod

TASKS = p1mod.PROBLEMS            # canonical 14-task order
METHODS = p1mod.METHODS           # canonical 17-method order
METHOD_PROBLEMS = p1mod.METHOD_PROBLEMS   # structural applicability

# Methods whose "default" arm is single-instance sgd (bs=1) rather than
# full-batch gd (mirrors METHOD_DEFAULT_OPT/BS in slurm/submit_bench_p1.sh).
BS1_METHODS = {"spo", "nce", "pointLTR", "pairLTR", "listLTR"}

# Paper symbols for the Phase-2 method-specific HPs. perturb's two sub-sweeps
# expose either sigma or n_samples as the winning hp_name; n_samples gets M to
# avoid colliding with lodl's K.
HP_SYMBOL = {
    "dflalpha":     ("α", r"\alpha"),
    "lambd":        ("λ", r"\lambda"),
    "tau":          ("τ", r"\tau"),
    "num_samples":  ("K", "K"),
    "sigma":        ("σ", r"\sigma"),
    "n_samples":    ("M", "M"),
    "stein_weight": ("w", "w"),
}

METHOD_MD = {
    "mse": "MSE", "mse_train": "MSE-tr", "mse_val": "MSE-val",
    "dfl": "DFL", "identity": "Identity", "spo": "SPO+", "nce": "NCE",
    "blackbox": "Blackbox", "pointLTR": "ptLTR", "pairLTR": "prLTR",
    "listLTR": "lsLTR", "lodl": "LODL", "perturb": "DPO", "pg": "PG",
    "qptl": "QPTL", "cpLayer": "cpLayer", "dad": "DAD",
}

METHOD_TEX = {
    "mse": "MSE", "mse_train": r"MSE\textsubscript{tr}",
    "mse_val": r"MSE\textsubscript{val}",
    "dfl": "DFL", "identity": "Identity",
    "spo": r"SPO\textsuperscript{+}", "nce": "NCE", "blackbox": "Blackbox",
    "pointLTR": "ptLTR", "pairLTR": "prLTR", "listLTR": "lsLTR",
    "lodl": "LODL", "perturb": "DPO", "pg": "PG", "qptl": "QPTL",
    "cpLayer": "cpLayer", "dad": "DAD",
}

TASK_MD = {
    "knapsack": "KS", "knapsack-real": "KS-E", "energy": "En",
    "budgetalloc": "BA", "cubic": "Cu", "bipartitematching": "BM",
    "portfolio": "Pf", "asurv": "AS", "cook_county": "CC",
    "speed_humps": "SH", "sp_synth": "SP-s", "sp_planted": "SP-p",
    "pg_misspec": "PG-ms", "shortestpath": "SP-W",
}

TASK_TEX = {
    "knapsack": "KS", "knapsack-real": "KS-E", "energy": "En",
    "budgetalloc": "BA", "cubic": "Cu", "bipartitematching": "BM",
    "portfolio": "Pf", "asurv": "AS", "cook_county": "CC",
    "speed_humps": "SH", "sp_synth": r"SP\textsubscript{s}",
    "sp_planted": r"SP\textsubscript{p}",
    "pg_misspec": r"PG\textsubscript{ms}", "shortestpath": "SP-W",
}

TASK_FULL = {
    "knapsack": "Knapsack (synth)", "knapsack-real": "Knapsack (energy)",
    "energy": "Energy scheduling", "budgetalloc": "Budget allocation",
    "cubic": "Cubic Top-K", "bipartitematching": "Bipartite matching",
    "portfolio": "Portfolio", "asurv": "Aerial survey (asurv)",
    "cook_county": "Cook County", "speed_humps": "Speed humps",
    "sp_synth": "Shortest path (synth)", "sp_planted": "Shortest path (planted)",
    "pg_misspec": "PG misspec", "shortestpath": "Warcraft shortest path",
}


def batch_meaning(label, method, task):
    """Resolve a batch-arm label to its real regime for (method, task).

    default = the original Geng et al. regime: full-batch gd, except bs=1 sgd
    for SPO/NCE/LTR. alt = minibatch sgd bs=32 (bs=4 on bipartitematching).
    """
    if label not in p1mod.BATCH_LABELS:
        raise ValueError(f"unknown batch label {label!r}")
    if label == "default":
        return "bs=1" if method in BS1_METHODS else "full batch"
    return "bs=4" if task == "bipartitematching" else "bs=32"


def structurally_absent(method, task):
    return method in METHOD_PROBLEMS and task not in METHOD_PROBLEMS[method]


def get_cell(p2, p1, method, task):
    """Best-config record for (method, task), or None.

    Prefers the Phase-2 file (which carries phase-1 fallback entries for
    methods without a Phase-2 sweep); falls back to the Phase-1 file only
    when the cell is absent from the Phase-2 one.
    """
    entry = p2.get(method, {}).get(task)
    if entry is None:
        entry = p1.get(method, {}).get(task)
    if entry is None or entry.get("lr") is None:
        return None
    return {
        "lr": entry["lr"],
        "batch": batch_meaning(entry["batch"], method, task),
        "hp_name": entry.get("hp_name"),
        "hp_value": entry.get("hp_value"),
    }


def cell_str(cell, tex=False):
    if cell is None:
        return "--"
    parts = [cell["lr"], "full" if cell["batch"] == "full batch" else cell["batch"]]
    if cell["hp_name"] is not None:
        uni, math = HP_SYMBOL[cell["hp_name"]]
        if tex:
            parts.append(f"${math}{{=}}{cell['hp_value']}$")
        else:
            parts.append(f"{uni}={cell['hp_value']}")
    return " / ".join(parts)


def write_csv(grid, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "lr", "batch", "hp_name", "hp_value"])
        for method in METHODS:
            for task in TASKS:
                cell = grid[method][task]
                if cell is None:
                    continue
                w.writerow([method, task, cell["lr"], cell["batch"],
                            cell["hp_name"] or "", cell["hp_value"] or ""])


def write_md(grid, path):
    lines = ["# Best hyperparameters per method × task", ""]
    lines.append(
        "Selection: minimum 10-seed-mean validation decision regret over the "
        "Phase-1 grid (5 learning rates × 2 batch regimes) plus the Phase-2 "
        "method-specific HP grid. Cell format: `lr / batch[ / hp=value]`."
    )
    lines.append("")
    header = ["Method"] + [TASK_MD[t] for t in TASKS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for method in METHODS:
        row = [METHOD_MD[method]] + [cell_str(grid[method][t]) for t in TASKS]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "**Legend**",
        "",
        "- Batch regimes: `full` = full-batch gradient descent; `bs=n` = "
        "minibatch SGD with batch size n (bs=4 only on bipartite matching).",
        "- HP symbols: α = DFL alpha (`dflalpha`), λ = Blackbox smoothing "
        "(`lambd`), τ = temperature (`tau`, QPTL/lsLTR), K = LODL samples "
        "(`num_samples`), σ = perturbation width (`sigma`, DPO/PG), M = DPO "
        "samples (`n_samples`), w = DAD Stein weight (`stein_weight`). "
        "Methods without a symbol have no Phase-2 HP sweep; their best "
        "config comes from Phase 1.",
        "- `--` = method not applicable (QPTL/cpLayer only on KS/BM/Pf; "
        "PG not on SP-W).",
        "- Tasks: " + ", ".join(f"{TASK_MD[t]} = {TASK_FULL[t]}" for t in TASKS)
        + ".",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def tex_subtable(grid, tasks):
    lines = [r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{@{}l" + "c" * len(tasks) + r"@{}}",
             r"\toprule",
             " & " + " & ".join(TASK_TEX[t] for t in tasks) + r" \\",
             r"\midrule"]
    for method in METHODS:
        cells = [cell_str(grid[method][t], tex=True) for t in tasks]
        lines.append(METHOD_TEX[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return lines


def write_tex(grid, path):
    caption = (
        "Best hyperparameters per method $\\times$ task. Each cell shows "
        "learning rate / batch regime / method-specific hyperparameter "
        "(where one is swept in Phase 2). Selection $=$ minimum 10-seed-mean "
        "validation decision regret from the Phase-1/2 sweep: Phase 1 sweeps "
        "5 learning rates $\\times$ 2 batch regimes per method $\\times$ task, "
        "Phase 2 sweeps the method-specific hyperparameter at the Phase-1-best "
        "(LR, batch). Batch regimes: full $=$ full-batch gradient descent; "
        "bs=$n$ $=$ minibatch SGD with batch size $n$. Symbols: $\\alpha$ "
        "(DFL), $\\lambda$ (Blackbox), $\\tau$ (QPTL, lsLTR), $K$ (LODL "
        "samples), $\\sigma$ (DPO/PG perturbation width), $M$ (DPO samples), "
        "$w$ (DAD Stein weight). `--' $=$ method not applicable. Tasks: "
        + ", ".join(f"{TASK_TEX[t]} = {TASK_FULL[t]}" for t in TASKS) + "."
    )
    lines = [r"\begin{table}[t]",
             r"\centering",
             rf"\caption{{{caption}}}",
             r"\label{tab:best-hyperparams}",
             r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.1}"]
    lines += tex_subtable(grid, TASKS[:7])
    lines.append(r"\vspace{0.75em}")
    lines.append("")
    lines += tex_subtable(grid, TASKS[7:])
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Emit the best-(LR, batch, HP) table from the sweep JSONs.")
    ap.add_argument("--p2_json", default="results/bench_p2_best_val.json",
                    help="Phase-2 val-selected best-config JSON (primary)")
    ap.add_argument("--p1_json", default="results/bench_p1_best_val.json",
                    help="Phase-1 val-selected best-config JSON (fallback)")
    ap.add_argument("--outdir", default="results/tables",
                    help="output directory for the .csv/.md/.tex files")
    args = ap.parse_args()

    with open(args.p2_json) as f:
        p2 = json.load(f)
    with open(args.p1_json) as f:
        p1 = json.load(f)

    grid = {}
    for method in METHODS:
        grid[method] = {}
        for task in TASKS:
            cell = get_cell(p2, p1, method, task)
            grid[method][task] = cell
            if cell is None and not structurally_absent(method, task):
                print(f"WARNING: no result for {method} x {task} "
                      "(not a structural absence)", file=sys.stderr)

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "best_hyperparams.csv")
    md_path = os.path.join(args.outdir, "best_hyperparams.md")
    tex_path = os.path.join(args.outdir, "best_hyperparams.tex")
    write_csv(grid, csv_path)
    write_md(grid, md_path)
    write_tex(grid, tex_path)

    n_cells = sum(1 for m in METHODS for t in TASKS if grid[m][t] is not None)
    print(f"Wrote {csv_path}, {md_path}, {tex_path} ({n_cells} cells)")


if __name__ == "__main__":
    main()
