"""
table_geng_repro.py
-------------------
Appendix "reproduction check" table: Geng et al. (NeurIPS 2024) published
Table-3 values vs. our re-run of their released code under their protocol
(their seed 2023, data, splits, 300 epochs, patience 40, default batch regime,
method hyperparameters at released defaults, learning rate validation-selected
over the same five-value grid). Columns: Geng et al. (1 run), ours seed0
(1 run), % difference, and ours 10-run mean +/- std over model-init seeds.

Reads  results/geng_reproduction_data.json  (per-cell Geng / seed0 / mean /
std / n / flags plus summary statistics; produced from the sweep output by the
research pipeline) and writes  results/tables/tabapp_geng_repro.tex , the
two-panel one-page table of the paper's appendix. It prints the summary
statistics quoted in the appendix text. No saved_records/ is needed.

Variant notes:
  * energy x {mse, dfl, identity} come from the val-regret re-run
    (bench_energyvr_*, solver evaluation ON, checkpoints selected on validation
    regret), so no selection-signal flag remains. See REPRODUCING.md
    "Selection signal".
  * cubic rows are the released-code (pre-correction) values, deliberately:
    the table checks reproduction of the published numbers, which carry the
    Cubic Top-K sense bug. See REPRODUCING.md "Cubic Top-K sense correction".

Usage (from the repo root):
    python experiments/table_geng_repro.py
    python experiments/table_geng_repro.py --json results/geng_reproduction_data.json \
        --out results/tables/tabapp_geng_repro.tex
"""

import argparse
import json
import os

PROB_PAPER = {
    "knapsack": "Knapsack (synth.)", "knapsack-real": "Knapsack (energy)",
    "energy": "Energy scheduling", "budgetalloc": "Budget allocation",
    "cubic": "Cubic Top-$K$", "bipartitematching": "Bipartite matching",
    "portfolio": "Portfolio (absolute regret)",
}

METHOD_LABELS = {
    "mse": "MSE", "dfl": "DFL", "blackbox": "Blackbox", "identity": "Identity",
    "spo": r"SPO\textsuperscript{+}", "nce": "NCE", "pointLTR": "pt-LTR",
    "pairLTR": "pr-LTR", "listLTR": "L-LTR", "lodl": "LODL",
    "perturb": "DPO", "cpLayer": "cpLayer", "qptl": "QPTL",
}

ABSOLUTE = {"portfolio"}  # absolute (not relative) regret, 3 decimals

PANELS = [["knapsack", "knapsack-real", "energy"],
          ["budgetalloc", "cubic", "bipartitematching", "portfolio"]]

CAPTION = r"""\caption{Reproduction check: published benchmark values of \citet{gengBenchmarkingPtOPnO2024} (their Table 3) vs.\ our re-run of the authors' released code under their protocol (matched data, splits, training settings and hyperparameter search). 
For our experiments, we report both a single \emph{1 run} with exactly matched seed (using seed in their released code), as well as \emph{mean $\pm$ std} of 10 runs from distinct initializations.
%: their seed, data, and splits, 300 epochs, patience 40, their batch regime (full batch, or batch size 1 for SPO\textsuperscript{+}/NCE/LTR), method hyperparameters at released defaults, and learning rate validation-selected over the same five-value grid. 
Reported values represent test relative regret (as a percentage), except for the Portfolio task, where we report absolute regret as in \citeauthor{gengBenchmarkingPtOPnO2024}'s Table 3.
To help understand relative differences between results, we report the \emph{percentage~difference} $=$ (ours $-$ theirs)$/|$theirs$| \cdot 100\%$.
\textbf{Special cases:} `--' means only 1 run exists, as on Energy scheduling due to runtime expense. $\dagger$: documented LODL configuration difference (Sec.~\ref{app:repro-lodl}).}"""

COLSPEC = r"""\begin{tabular}[t]{l 
    >{\raggedleft\arraybackslash}p{15mm} 
    >{\raggedleft\arraybackslash}p{10mm} 
    >{\raggedleft\arraybackslash}p{10mm} 
    >{\raggedleft\arraybackslash}p{18mm} 
    }"""

# The two panels differ by one character in the header ("%~diff." vs "%-diff.").
HEADERS = [
    r"""Method & {Geng et al.\newline1 run} 
    & {ours\newline1 run} & \%~diff. 
    & {ours 10 run \newline mean $\pm$ std} \\""",
    r"""Method & {Geng et al.\newline1 run} 
    & {ours\newline1 run} & \%-diff. 
    & {ours 10 run \newline mean $\pm$ std} \\""",
]


def fmt(v, prob):
    if v is None:
        return "--"
    d = 3 if prob in ABSOLUTE else (2 if abs(v) < 100 else 1)
    return f"{v:.{d}f}"


def rel(g, o):
    return None if (g is None or o is None or g == 0) else (o - g) / abs(g) * 100.0


def panel(probs, by_prob, header):
    lines = [COLSPEC, r"\toprule", header, r"\midrule"]
    for i, prob in enumerate(probs):
        if i:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{PROB_PAPER[prob]}}}}} \\")
        for r in by_prob[prob]:
            geng, s0, mean, std, n = (r["geng"], r["seed0"], r["mean"],
                                      r["std"], r["n"])
            r0 = rel(geng, s0)
            tex_flags = r["flags"].replace("†", r"$^\dagger$").replace("‡", r"$^\ddagger$")
            if mean is None or n <= 1:
                ms = "--"
            elif std is not None:
                ms = f"{fmt(mean, prob)} $\\pm$ {fmt(std, prob)}"
            else:
                ms = fmt(mean, prob)
            lines.append(
                f"{METHOD_LABELS.get(r['method'], r['method'])}{tex_flags} & "
                f"{fmt(geng, prob)} & {fmt(s0, prob)} & "
                f"{'--' if r0 is None else f'{r0:+.1f}'} & {ms} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="results/geng_reproduction_data.json")
    ap.add_argument("--out", default="results/tables/tabapp_geng_repro.tex")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    meta, st = data["meta"], data["stats"]

    # ---- summary statistics ----
    print(f"cells compared (seed0): {st['n_cells']}")
    print(f"median |rel diff|: {st['median_abs_rel_diff']:.1f}%   "
          f"within 5%: {st['within_5pct']:.0f}%   "
          f"10%: {st['within_10pct']:.0f}%   20%: {st['within_20pct']:.0f}%")
    print(f"excluding flagged cells (n={st['n_unflagged']}): "
          f"median {st['unflagged_median_abs_rel_diff']:.1f}%   "
          f"within 10%: {st['unflagged_within_10pct']:.0f}%   "
          f"20%: {st['unflagged_within_20pct']:.0f}%")
    print(f"relative-regret cells within 2 regret points: "
          f"{st['rel_regret_within_2pts']}/{st['rel_regret_cells']}")
    print(f"(variant: energyvr={meta['energyvr']}, cubicfix={meta['cubicfix']})")

    by_prob = {}
    for r in data["rows"]:
        by_prob.setdefault(r["problem"], []).append(r)

    lines = [
        r"% Auto-generated by rethink_exp/table_geng_repro.py --paper — do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        CAPTION,
        r"\label{tab:geng-reproduction}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\footnotesize",
    ]
    lines += panel(PANELS[0], by_prob, HEADERS[0])
    lines.append(r"\hfill")
    lines += panel(PANELS[1], by_prob, HEADERS[1])
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
