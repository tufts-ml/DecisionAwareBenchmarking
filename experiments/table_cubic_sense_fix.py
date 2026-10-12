"""
table_cubic_sense_fix.py
------------------------
Appendix table for the Cubic Top-K task declaration: per method, Geng et al.'s
published relative regret, this code under the upstream declaration
(MINIMIZE), and this code with MAXIMIZE declared.

Inputs  (results/):
  cubic_upstream_sense.json   method -> cubic cell under the upstream
                              declaration (10-seed mean +/- std test relative
                              regret; same fields as a bench_p2_best_val.json cell)
  bench_p2_best_val.json      val-selected picks (MAXIMIZE declaration)
Also experiments/bench_gains_common.GENG (published values).
DPO and LODL are sense-independent (bit-identical runs) and MSE / pointwise
LTR never read the sense, so their MAXIMIZE column repeats the upstream value.
Output (results/tables/): tabapp_cubic_sense_fix.tex

Run from the repo root::

    PYTHONPATH=$PWD python experiments/table_cubic_sense_fix.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_gains_common as common  # noqa: E402

ROWS = [  # (key, paper name, sense-dependence note)
    ("mse",      "MSE",            "no"),
    ("dfl",      "DFL",            "self-objective$^{\\dagger}$"),
    ("identity", "Identity",       "self-objective$^{\\dagger}$"),
    ("blackbox", "Blackbox",       "self-objective$^{\\dagger}$"),
    ("spo",      "SPO\\textsuperscript{+}", "inverted"),
    ("nce",      "NCE",            "inverted"),
    ("pointLTR", "pt-LTR",          "no"),
    ("pairLTR",  "pr-LTR",          "sign-sensitive"),
    ("listLTR",  "L-LTR",          "sign-sensitive"),
    ("lodl",     "LODL",           "no$^{\\ddagger}$"),
    ("perturb",  "DPO",            "no$^{\\ddagger}$"),
    ("pg",       "PG",             "self-objective$^{\\dagger}$"),
]


def cell(e):
    if not e or e.get("test") is None:
        return "--", None
    t, s, n = 100 * e["test"], 100 * (e.get("test_std") or 0.0), e.get("test_n_done", 1)
    return (f"{t:.2f} $\\pm$ {s:.2f}" if n > 1 else f"{t:.2f}"), t


def main():
    upstream = json.load(open("results/cubic_upstream_sense.json"))
    new = json.load(open("results/bench_p2_best_val.json"))
    lines = []
    for key, name, dep in ROWS:
        geng = common.GENG.get(key, {}).get("cubic")
        g = f"{geng:.2f}" if geng is not None else "--"
        oe = upstream.get(key, {})
        s0 = oe.get("test_seed_values", {}).get("0")
        s0 = f"{100 * s0:.2f}" if s0 is not None else "--"
        o, ov = cell(oe)
        c, cv = cell(new.get(key, {}).get("cubic"))
        changed = cv is not None and (ov is None or abs(cv - ov) > 1e-9)
        cc = f"\\textbf{{{c}}}" if changed else c
        lines.append(f"{name} & {dep} & {g} & {s0} & {o} & {cc} \\\\")
    tex = r"""\begin{table}[!h]
\centering
\small
\caption{Cubic Top-$K$ after correcting the problem's declared optimisation sense.
The released \texttt{CubicTopK} problem declares a \emph{minimisation} sense while its
top-$K$ solver selects the $K$ \emph{largest} predicted values, so every surrogate
loss that branches on the declared sense trained with the wrong sign on this task;
the released code reproduces the published numbers (middle columns) bug included.
Columns: published value from \citet{gengBenchmarkingPtOPnO2024}; our run of the
released code (seed 0 reproduces the published single-run values of the affected methods to the decimal; the 10-seed mean also shows the spread across initialisations); and our re-run with the one-line fix under the identical protocol
(two-phase search, 10 initialisation seeds; bold = re-run cells).
Test relative regret (\%), 10-seed mean $\pm$ std.
$^{\dagger}$Decision term is the self-objective $\langle\hat{y}, z(\hat{y})\rangle$,
whose gradient never involves the true costs; with the correct sign it reinforces the
initial ranking, with the inverted sign it churned it and checkpoint selection on
validation regret harvested the best moment. $^{\ddagger}$Sense-independent in code
(DPO's loss uses $|\cdot|$; LODL reads the sense only to choose among solver restarts);
re-runs are bit-identical, so the original value stands.}
\label{tab:cubic-sense-fix}
\begin{tabular}{l l r r r r}
\toprule
Method & Sense use & Published & \multicolumn{2}{c}{Released code} & Corrected \\
 & & (single run) & seed 0 & 10-seed mean $\pm$ std & 10-seed mean $\pm$ std \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/tabapp_cubic_sense_fix.tex", "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
