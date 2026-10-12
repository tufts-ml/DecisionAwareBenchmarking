"""
table_fullgrid_paper.py
-----------------------
Appendix tables for the full joint-grid check of the two-phase tuning design
(App. "Joint hyperparameter grid check").

Inputs  (results/):
  bench_fullgrid_best_val.json   joint-grid results for all 8 swept pairs
  bench_p2_best_val.json         the paper-current two-stage picks (per-task
                                 ranking table)
Outputs (results/tables/), paper notation (b = batch size, eta = learning
rate, method HP symbols as in tab_hyperparam_candidates.tex):
  tabapp_fullgrid_pairs.tex   one row per swept (method, task) pair:
                              two-stage pick vs joint-grid val-argmax,
                              10-seed mean test relative regret for both,
                              and the relative change
  tabapp_fullgrid_cubic.tex   the Cubic Top-K column of the paper's 14
                              methods with the joint-grid values swapped in,
                              and the resulting rank change

Run from the repo root::

    PYTHONPATH=$PWD python experiments/table_fullgrid_paper.py
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_runtime_paper import METHODS, METHOD_LABELS  # noqa: E402

FG_JSON = "results/bench_fullgrid_best_val.json"
P2_JSON = "results/bench_p2_best_val.json"

# Paper task names for the swept tasks.
TASK_FULL = {
    "knapsack": "Knapsack (synth.)",
    "pg_misspec": "PG Misspecified",
    "cubic": "Cubic Top-$K$",
}
# Row order: wave 1 (inert HP axes) then wave 2 (active HP axes on cubic).
PAIR_ORDER = [
    ("listLTR", "knapsack"), ("blackbox", "knapsack"),
    ("listLTR", "pg_misspec"), ("blackbox", "pg_misspec"),
    ("listLTR", "cubic"), ("dfl", "cubic"), ("lodl", "cubic"), ("pg", "cubic"),
]

# Method-HP symbol per method, as in tab_hyperparam_candidates.tex.
HP_SYMBOL = {
    "dfl": r"\alpha", "blackbox": r"\lambda", "listLTR": r"\tau",
    "lodl": "K", "perturb": r"\sigma", "pg": "h",
}
# Methods whose "default" batch arm is batch size 1 rather than full batch
# (submit_bench_p1.sh METHOD_DEFAULT_BS).
BS1_DEFAULT = {"spo", "nce", "pointLTR", "pairLTR", "listLTR"}

# hp_tag prefixes -> numeric value string.
TAG_RE = re.compile(r"^(tau|lam|alpha|ns|sw|s)(.+)$")


def hp_value_from_tag(tag):
    """'tau0.1' -> '0.1'; 'alpha10.0' -> '10'; 's0p5' -> '0.5'; 'ns250' -> '250'."""
    if tag is None:
        return None
    m = TAG_RE.match(tag)
    raw = m.group(2) if m else tag
    raw = raw.replace("p", ".")
    try:
        v = float(raw)
    except ValueError:
        return raw
    return fmt_num(v)


def fmt_num(v):
    v = float(v)
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def fmt_batch(method, batch):
    if batch == "alt":
        return "32"
    return "1" if method in BS1_DEFAULT else r"\mathrm{all}"


def fmt_lr(lr):
    # '1e-2' -> '10^{-2}', '5e-3' -> '5{\times}10^{-3}'
    mant, exp = lr.split("e")
    exp = int(exp)
    if mant == "1":
        return rf"10^{{{exp}}}"
    return rf"{mant}{{\times}}10^{{{exp}}}"


def fmt_config(method, batch, lr, hp_value):
    sym = HP_SYMBOL[method]
    return rf"$b{{=}}{fmt_batch(method, batch)},\ \eta{{=}}{fmt_lr(lr)},\ {sym}{{=}}{hp_value}$"


def fmt_pct(x, std=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{100*x:.2f}"
    if std is not None:
        s += rf" $\pm$ {100*std:.2f}"
    return s


# ---- Table 1: per-pair comparison ------------------------------------------

def pairs_table(fg):
    rows = []
    for method, task in PAIR_ORDER:
        e = fg[method][task]
        fb, ts, cmp_ = e["fullgrid_best"], e["two_stage"], e["compare"]
        ts_hp = hp_value_from_tag(ts["hp_tag"])
        fg_hp = fmt_num(fb["hp_value"]) if fb.get("hp_value") is not None \
            else hp_value_from_tag(fb["hp_tag"])
        same = cmp_["same_config"]
        delta = (ts["test"] - fb["test"]) / abs(ts["test"]) * 100 if ts["test"] else float("nan")
        delta_s = "0.0" if same else f"{delta:+.1f}"
        fg_cfg = "same" if same else fmt_config(method, fb["batch"], fb["lr"], fg_hp)
        assert e["grid_n_cells_full_seeds"] == e["grid_n_cells"] == 50, (method, task)
        rows.append(
            f"{METHOD_LABELS[method]} & {TASK_FULL[task]} & "
            f"{fmt_config(method, ts['batch'], ts['lr'], ts_hp)} & {fg_cfg} & "
            f"{fmt_pct(ts['test'])} & {fmt_pct(fb['test'], fb.get('test_std'))} & {delta_s} \\\\"
        )
    caption = (
        r"Two-phase tuning vs.\ the full joint grid. For eight method--task pairs we trained "
        r"every cell of the joint grid $\{$2 batch sizes$\}\times\{$5 learning rates$\}\times"
        r"\{$5 method-specific values$\}$ (Tab.~\ref{tab:methods-hyperparams}), 10 initializations "
        r"per cell (500 runs per pair; every one of the 50 configurations is complete at 10/10 runs). "
        r"\emph{Two-phase pick} is the configuration selected by the paper's protocol "
        r"(Sec.~\ref{sec:bench_replication}); \emph{joint-grid pick} is the configuration with the "
        r"lowest 10-run mean validation regret over the whole grid (``same'' when both coincide). "
        r"Test columns are 10-run mean test relative regret (\%) at each pick ($\pm$ std over "
        r"runs for the joint grid); $\Delta$ $=$ (two-phase $-$ joint)/two-phase $\times 100$, positive "
        r"when the joint grid found a better configuration. Selection is validation-only throughout. "
        r"The first four rows are tasks on which the method-specific axis is inert (see text); the "
        r"last four are the Cubic Top-$K$ pairs whose method-specific hyperparameter is active."
    )
    lines = [
        "% Auto-generated by rethink_exp/table_fullgrid_paper.py -- do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:fullgrid-pairs}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\footnotesize",
        r"\begin{tabular}{@{}llllrrr@{}}",
        r"\toprule",
        r"Method & Task & Two-phase pick & Joint-grid pick & \makecell[r]{Test RR (\%)\\two-phase} & \makecell[r]{Test RR (\%)\\joint grid} & $\Delta$ (\%) \\",
        r"\midrule",
    ]
    lines += rows[:4] + [r"\addlinespace"] + rows[4:]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---- Table 2: cubic column with joint-grid values swapped in ----------------

def avg_ranks(values):
    """Average ranks (1 = lowest) with ties averaged; NaN -> None."""
    idx = [i for i, v in enumerate(values) if v is not None and not np.isnan(v)]
    vals = np.array([values[i] for i in idx])
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(len(vals))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    out = [None] * len(values)
    for r, k in zip(ranks, idx):
        out[k] = r
    return out


def fmt_rank(r):
    if r is None:
        return "--"
    return f"{r:g}" if r == int(r) else f"{r:.1f}"


def cubic_table(fg, p2, task="cubic"):
    before, after, changed = [], [], []
    for m in METHODS:
        e = p2.get(m, {}).get(task)
        t = e["test"] if e and e.get("test") is not None else float("nan")
        before.append(t)
        if m in fg and task in fg[m]:
            after.append(fg[m][task]["fullgrid_best"]["test"])
            changed.append(True)
        else:
            after.append(t)
            changed.append(False)
    rb, ra = avg_ranks(before), avg_ranks(after)
    equiv = {"mse", "mse_train", "mse_val", "identity", "spo", "nce",
             "blackbox", "pointLTR", "pairLTR"}
    rows = []
    for m, b, a, c, r0, r1 in zip(METHODS, before, after, changed, rb, ra):
        lbl = METHOD_LABELS[m]
        if c:
            rows.append(
                f"{lbl} & {fmt_pct(b)} & \\jointfresh{{{fmt_pct(a)}}} & {fmt_rank(r0)} & "
                f"\\jointfresh{{{fmt_rank(r1)}}} \\\\"
            )
        elif m in equiv:
            rows.append(f"{lbl} & {fmt_pct(b)} & \\jointequiv{{{fmt_pct(a)}}} & {fmt_rank(r0)} & {fmt_rank(r1)} \\\\")
        else:
            rows.append(f"{lbl} & {fmt_pct(b)} & {fmt_pct(a)} & {fmt_rank(r0)} & {fmt_rank(r1)} \\\\")
    caption = r"""Cubic Top-$K$ test relative regret (\%, 10-run mean) and within-task rank for the paper's 14 methods under the two-phase protocol (as reported in Fig.~\ref{fig:bench_bump}) and the joint-grid one-phase protocol for select methods.
We mark methods in \jointfresh{bold orange} if their results come from the joint grid experiments in Tab.~\ref{tab:fullgrid-pairs}. We use \jointequiv{bold blue} for methods without a method-specific hyperparameter (MSE variants, Identity, SPO\textsuperscript{+}, NCE, pt-LTR, pr-LTR), where their two-phase results are already equivalent to the joint-grid protocol. 
Ranks are 1 $=$ best; ties share the average rank."""
    lines = [
        "% Auto-generated by rethink_exp/table_fullgrid_paper.py -- do not edit.",
        r"\newcommand{\jointfresh}[1]{{\color{orange} \textbf{#1}}}",
        r"\newcommand{\jointequiv}[1]{{\color{blue} \textbf{#1}}}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:fullgrid-cubic}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & \makecell[r]{Test RR (\%)\\two-phase} & \makecell[r]{Test RR (\%)\\+ joint grid} & \makecell[r]{Rank\\two-phase} & \makecell[r]{Rank\\+ joint grid} \\",
        r"\midrule",
    ] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    # Console summary for the appendix prose.
    print(f"-- {task}: rank changes (two-phase -> +joint grid)")
    for m, r0, r1, c in zip(METHODS, rb, ra, changed):
        flag = " *" if c else ""
        if r0 != r1 or c:
            print(f"   {m:10s} {fmt_rank(r0):>4s} -> {fmt_rank(r1):>4s}{flag}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg_json", default=FG_JSON)
    ap.add_argument("--p2_json", default=P2_JSON)
    ap.add_argument("--out_dir", default="results/tables")
    args = ap.parse_args()

    fg = json.load(open(args.fg_json))
    p2 = json.load(open(args.p2_json))
    missing = [(m, t) for m, t in PAIR_ORDER if m not in fg or t not in fg[m]]
    if missing:
        sys.exit(f"{args.fg_json} lacks pairs {missing}.")

    os.makedirs(args.out_dir, exist_ok=True)
    p1 = os.path.join(args.out_dir, "tabapp_fullgrid_pairs.tex")
    p2path = os.path.join(args.out_dir, "tabapp_fullgrid_cubic.tex")
    with open(p1, "w") as f:
        f.write(pairs_table(fg))
    with open(p2path, "w") as f:
        f.write(cubic_table(fg, p2))
    print(f"wrote {p1}\nwrote {p2path}")

    # Headline numbers for the prose.
    print("-- per-pair summary")
    for m, t in PAIR_ORDER:
        e = fg[m][t]
        ts, fb = e["two_stage"]["test"], e["fullgrid_best"]["test"]
        print(f"   {m:9s} {t:11s} 2S={100*ts:7.3f}%  FG={100*fb:7.3f}%  "
              f"same={e['compare']['same_config']}  ratio={ts/fb if fb else float('nan'):.2f}x")


if __name__ == "__main__":
    main()
