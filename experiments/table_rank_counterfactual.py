"""
table_rank_counterfactual.py
----------------------------
"Does the expanded sweep explain MSE's standing?" -- tuning-effect tables
restricted to the task subset **not recommended** in the paper:

    knapsack (gen), knapsack (real), energy, cubic, bipartite matching,
    shortestpath (warcraft 12x12)

(the same TASKS list as ``fig_bench_bump_not_recommended.py``).

Two protocols are compared per (method, task):

  A "original"   -- the original benchmark's tuning regime: default batch arm
                    (full-batch ``gd``, or ``bs=1`` for SPO/NCE/LTR), learning
                    rate val-selected over the 5-LR grid, method HP left at its
                    default.
  B "full sweep" -- our regime: val-best over {both batch arms x 5 LRs} union
                    the Phase-2 method-HP grid.

Reported per method, aggregated over the subset:

  * change in relative regret   dRR% = (A - B) / |A| * 100   (positive = better)
  * change in rank              dRank = rank_A - rank_B      (positive = better)

Ranks are computed per task over the methods that have a number under *both*
protocols, so the two rankings always compare the same pool. Ties get average
ranks.

All selections are by validation signal (no test leakage); the reported numbers
are test relative regret. Three seed-aggregation modes, as in the other gain
tables: ``seed0`` / ``bestseed`` / ``mean``.

Outputs (per mode)::

    results/tables/notrec_regret_change_{seed0,bestseed,mean}.tex
    results/tables/notrec_rank_change_{seed0,bestseed,mean}.tex

Usage::

    python experiments/table_rank_counterfactual.py                  # all 3 modes
    python experiments/table_rank_counterfactual.py --modes mean
    python experiments/table_rank_counterfactual.py --subset all     # all 14 tasks
    python experiments/table_rank_counterfactual.py --use_cache      # reuse last scan
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_gains_common as G

# ---- Task subsets -----------------------------------------------------------

NOT_RECOMMENDED = [
    "knapsack", "knapsack-real", "energy", "cubic",
    "bipartitematching", "shortestpath",
]
# Canonical groups, matching fig_bench_bump_recommended.py's TASK_GROUPS.
RECOMMENDED_MAIN = ["budgetalloc", "pg_misspec", "sp_synth"]
GEOSPATIAL = ["cook_county", "speed_humps", "asurv"]

# Column groups for the wide variants: the not-recommended tasks plus the
# recommended and geospatial groups. Two flavours -- with and without warcraft
# (SP-W), whose full-batch arm is degenerate/incomplete so four methods have no
# protocol-A value there. Both leave out portfolio and sp_planted, which belong
# to neither added group.
def _groups(with_spw):
    nr = NOT_RECOMMENDED if with_spw else [t for t in NOT_RECOMMENDED
                                           if t != "shortestpath"]
    return [("Not recommended", list(nr)),
            ("Recommended", RECOMMENDED_MAIN),
            ("Geospatial", GEOSPATIAL)]


SUBSET_GROUPS = {
    "notrec_rec_geo": _groups(with_spw=False),       # 11 tasks
    "notrec_rec_geo_spw": _groups(with_spw=True),    # 12 tasks
}

SUBSETS = {
    "not_recommended": NOT_RECOMMENDED,
    "recommended": [p for p in G.PROBLEMS if p not in NOT_RECOMMENDED],
    "all": list(G.PROBLEMS),
    **{k: [t for _, grp in v for t in grp] for k, v in SUBSET_GROUPS.items()},
}
SUBSET_TAG = {"not_recommended": "notrec", "recommended": "rec", "all": "all",
              "notrec_rec_geo": "notrec-rec-geo",
              "notrec_rec_geo_spw": "notrec-rec-geo-spw"}
_NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def subset_desc(subset, tasks):
    """Caption phrase for the task set, honouring --drop_tasks."""
    n = _NUMWORD.get(len(tasks), str(len(tasks)))
    if subset == "not_recommended":
        return f"the {n} tasks \\emph{{not}} recommended in the paper"
    if subset == "recommended":
        return "the tasks recommended in the paper"
    if subset in SUBSET_GROUPS:
        spw = "" if "shortestpath" in tasks else " (warcraft dropped)"
        return (f"{n} tasks: the not-recommended subset{spw}, "
                "the recommended subset, and the geospatial subset")
    return f"all {n} benchmark tasks"


def groups_for(subset, tasks):
    """Column groups (label, tasks) for the header, or None for a flat header."""
    groups = SUBSET_GROUPS.get(subset)
    if not groups:
        return None
    out = [(lab, [t for t in grp if t in tasks]) for lab, grp in groups]
    return [(lab, grp) for lab, grp in out if grp] or None

# Same exclusions as fig_bench_bump_not_recommended.py: DAD is missing seeds on
# several of these tasks, and QPTL/cpLayer only apply to 2 of the 6.
DEFAULT_EXCLUDE = {"dad", "qptl", "cpLayer"}

MODE_FILETAG = {"seed0": "seed0", "best": "bestseed", "mean": "mean"}
MODE_LABEL = {"seed0": "init seed 0", "best": "best-validating seed",
              "mean": "10-seed mean"}

CACHE = "results/rank_counterfactual_cache.json"


# ---- Data -------------------------------------------------------------------

def compute_task(prob, methods, mode):
    """{method: {"A": pct|None, "B": pct|None}} for one (task, seed mode)."""
    out = {prob: {}}
    for _ in (0,):
        for method in methods:
            if not G.applies(method, prob):
                continue
            sel = G.p1_selection(prob, method, mode)
            fb, overall = sel["fb"], sel["overall"]
            A = fb[1] if fb else None

            cands = []
            if overall is not None and overall[1] is not None:
                cands.append((overall[0], overall[1]))
            lr2, b2 = G.p1_best_config(prob, method)
            if lr2 is not None:
                p2 = G.p2_selection(prob, method, mode, lr2, b2)
                if p2 and p2["tuned"] and p2["tuned"][1] is not None:
                    cands.append((p2["tuned"][0], p2["tuned"][1]))
            B = min(cands, key=lambda c: c[0])[1] if cands else None

            out[prob][method] = {"A": G.to_pct(A, prob), "B": G.to_pct(B, prob)}
        print(f"    [{mode}] {prob}: {len(out[prob])} methods", flush=True)
    return out[prob]


def rank_map(rows, key):
    """Average-tie ranks (1 = best) over methods with values under both A and B."""
    d = {m: r[key] for m, r in rows.items()
         if r["A"] is not None and r["B"] is not None}
    if not d:
        return {}
    order = sorted(d, key=lambda m: d[m])
    out, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and d[order[j + 1]] == d[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


# ---- Formatting -------------------------------------------------------------

def shade(v, lo=5.0, hi=25.0):
    """LaTeX cell colour for a signed improvement (positive = better)."""
    if v is None or abs(v) < lo:
        return ""
    inten = 18 if abs(v) < hi else 28
    return f"\\cellcolor{{{'green' if v > 0 else 'red'}!{inten}}}"


def fmt_pct(v):
    """Math mode so the sign renders as a true +/- glyph, not a hyphen."""
    if v is None:
        return "--"
    if abs(v) >= 1000:
        return f"${v:+.0f}$"
    return f"${v:+.1f}$"


def fmt_rank(v):
    return "--" if v is None else f"{v:.2f}"


def tex_escape_label(m):
    return G.METHOD_LABELS.get(m, m)


def tie_note(tasks, methods, data):
    """Flag tasks where >=3 methods share one A value (degenerate baseline).

    On these the rank-A values are tie-averaged, which materially changes how a
    rank *change* should be read -- so it goes in the caption rather than being
    left silent.
    """
    parts = []
    for p in tasks:
        groups = {}
        for m in methods:
            r = data.get(p, {}).get(m)
            if not r or r["A"] is None or r["B"] is None:
                continue
            groups.setdefault(round(r["A"], 6), []).append(m)
        if not groups:
            continue
        val, members = max(groups.items(), key=lambda kv: len(kv[1]))
        if len(members) >= 3:
            parts.append(f"{G.PROB_LABELS[p]} ({len(members)} methods at "
                         f"{val:.3g}\\%)")
    if not parts:
        return ""
    return (r" \textbf{Note:} under $A$ several methods collapse to an "
            r"identical degenerate value (full-batch training gives them no "
            r"effective signal), so their $A$ ranks are tie-averaged: "
            + "; ".join(parts) + ".")


PROTOCOL_SENTENCE = (
    r"$A$ is the original tuning regime (default batch arm, LR val-selected, "
    r"default method HP) and $B$ is our full sweep (both batch arms $\times$ "
    r"5 LRs, plus the Phase-2 method HP)")


def task_legend(tasks):
    """Expand the column abbreviations, for tables that show per-task columns."""
    parts = [f"{G.PROB_LABELS[p]} $=$ {G.PROB_FULL[p]}" for p in tasks]
    return r" Tasks: " + ", ".join(parts) + "."


def dvals(tasks, methods_data, m):
    """dRR% for method m over `tasks`, skipping tasks with no number."""
    out = []
    for p in tasks:
        r = methods_data.get(p, {}).get(m)
        v = G.rel_gain(r["A"], r["B"]) if r else None
        if v is not None:
            out.append(v)
    return out


def group_header(tasks, subset, n_trailing):
    """Optional 'Not recommended | Recommended | Geospatial' spanning header.

    Returns a list of extra header lines (possibly empty). Column 1 is the
    method label; the trailing summary columns are left unlabelled.
    """
    groups = groups_for(subset, tasks)
    if not groups:
        return []
    cells, rules, col = [""], [], 2
    for lab, grp in groups:
        cells.append(r"\multicolumn{" + str(len(grp)) + r"}{c}{" + lab + "}")
        rules.append(r"\cmidrule(lr){" + f"{col}-{col + len(grp) - 1}" + "}")
        col += len(grp)
    cells += [""] * n_trailing
    return [" & ".join(cells) + r" \\", "".join(rules)]


def regret_table(tasks, methods, data, mode, subset, short=False, lsfx=""):
    """Per-task dRR% grid + mean / median / helped."""
    cols = "@{}l" + "r" * len(tasks) + "rrr@{}"
    head = " & ".join([""] + [G.PROB_LABELS[p] for p in tasks]
                      + ["Mean", "Median", "helped"])
    ghead = group_header(tasks, subset, 3)
    if short:
        caption = (r"Change in test relative regret from the expanded sweep: "
                   r"$\Delta$RR\% $=(A-B)/|A|\times100$ (positive $=$ the sweep "
                   r"helped), where " + PROTOCOL_SENTENCE
                   + r"; values are the " + MODE_LABEL[mode] + "."
                   + task_legend(tasks))
    else:
        caption = (
            r"Change in test relative regret from the expanded sweep, over "
            + subset_desc(subset, tasks) + r" (\textbf{" + MODE_LABEL[mode] + r"}). "
            r"$\Delta$RR\% $=(A-B)/|A|\times100$, where " + PROTOCOL_SENTENCE
            + r". Positive $=$ the expanded sweep helped. All selections by "
            r"validation signal. `helped' counts tasks with $\Delta$RR${>}0$. "
            r"Cells shaded at $|\Delta|\geq5\%$ (darker at $\geq25\%$). Relative "
            r"changes on tasks with near-zero regret (Cu) and on degenerate "
            r"full-batch baselines (SP-W) are very large; the median column is "
            r"the robust summary." + tie_note(tasks, methods, data)
            + task_legend(tasks))
    lines = [
        r"\begin{table}[t]", r"\centering",
        *([r"\small"] if len(tasks) > 8 else []),
        r"\caption{" + caption + "}",
        r"\label{tab:" + SUBSET_TAG[subset] + "-regret-change-"
        + MODE_FILETAG[mode] + lsfx.replace("_", "-") + "}",
        r"\setlength{\tabcolsep}{" + ("3pt" if len(tasks) > 8 else "4pt") + "}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{" + cols + "}", r"\toprule",
        *ghead,
        head + r" \\", r"\midrule",
    ]
    for m in methods:
        vals, cells = [], []
        for p in tasks:
            r = data.get(p, {}).get(m)
            v = G.rel_gain(r["A"], r["B"]) if r else None
            if v is not None:
                vals.append(v)
            cells.append(f"{shade(v)}{fmt_pct(v)}")
        if not vals:
            continue
        helped = sum(1 for v in vals if v > 0)
        mean, med = float(np.mean(vals)), float(np.median(vals))
        lines.append(" & ".join(
            [tex_escape_label(m)] + cells
            + [f"{shade(mean)}{fmt_pct(mean)}", f"{shade(med)}{fmt_pct(med)}",
               f"{helped}/{len(vals)}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def group_summary_table(tasks, methods, data, mode, subset, short=False, lsfx=""):
    """Compact variant: mean / median / helped per task CATEGORY.

    No per-task columns, so the category membership goes in the caption instead
    of an abbreviation legend. Returns None for ungrouped subsets.
    """
    groups = groups_for(subset, tasks)
    if not groups:
        return None
    cols = "@{}l" + "rrc" * len(groups) + "@{}"
    g1, rules, col = [""], [], 2
    for lab, _ in groups:
        g1.append(r"\multicolumn{3}{c}{" + lab + "}")
        rules.append(r"\cmidrule(lr){" + f"{col}-{col + 2}" + "}")
        col += 3
    head = " & ".join([""] + ["Mean", "Median", "helped"] * len(groups))
    memb = "; ".join(f"\\emph{{{lab}}} $=$ "
                     + ", ".join(G.PROB_FULL[p] for p in grp)
                     for lab, grp in groups)
    metric = (r"Change in test relative regret from the expanded sweep, "
              r"aggregated within each task category: $\Delta$RR\% "
              r"$=(A-B)/|A|\times100$ (positive $=$ the sweep helped), where "
              + PROTOCOL_SENTENCE + r"; values are the " + MODE_LABEL[mode]
              + ".")
    if short:
        caption = metric + " Categories --- " + memb + "."
    else:
        caption = (metric + r" `helped' counts the tasks in that category with "
                   r"$\Delta$RR${>}0$; its denominator shows how many of them "
                   r"the method has a number for under both protocols. Means "
                   r"are unweighted and can be carried by a single task, so the "
                   r"median is the robust column. Categories --- " + memb + ".")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:" + SUBSET_TAG[subset] + "-regret-change-by-group-"
        + MODE_FILETAG[mode] + lsfx.replace("_", "-") + "}",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{" + cols + "}", r"\toprule",
        " & ".join(g1) + r" \\", "".join(rules),
        head + r" \\", r"\midrule",
    ]
    for m in methods:
        if not dvals(tasks, data, m):
            continue
        cells = []
        for _, grp in groups:
            vals = dvals(grp, data, m)
            if not vals:
                cells += ["--", "--", "--"]
                continue
            mean, med = float(np.mean(vals)), float(np.median(vals))
            cells += [f"{shade(mean)}{fmt_pct(mean)}",
                      f"{shade(med)}{fmt_pct(med)}",
                      f"{sum(v > 0 for v in vals)}/{len(vals)}"]
        lines.append(" & ".join([tex_escape_label(m)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def rank_table(tasks, methods, data, ranks, mode, subset, short=False, lsfx=""):
    """Per-task rank A->B + average rank under each protocol."""
    cols = "@{}l" + "c" * len(tasks) + "rrr@{}"
    head = " & ".join([""] + [G.PROB_LABELS[p] for p in tasks]
                      + [r"Rank$_A$", r"Rank$_B$", r"$\Delta$Rank"])
    ghead = group_header(tasks, subset, 3)
    pool = {p: len(ranks[p]["A"]) for p in tasks if ranks[p]["A"]}
    pool_str = ", ".join(f"{G.PROB_LABELS[p]}~{n}" for p, n in pool.items())
    if short:
        caption = (r"Change in rank from the expanded sweep: $\Delta$Rank $=$ "
                   r"Rank$_A-$Rank$_B$ (positive $=$ moved up), where "
                   + PROTOCOL_SENTENCE + r"; per-task cells give the method's "
                   r"rank$_A\rightarrow$rank$_B$ (1 $=$ best) among the methods "
                   r"shown, and values are the " + MODE_LABEL[mode] + "."
                   + task_legend(tasks))
    else:
        caption = (
            r"Change in rank from the expanded sweep, over " + subset_desc(subset, tasks)
            + r" (\textbf{" + MODE_LABEL[mode] + r"}). Per-task cells give the "
            r"method's rank under the original tuning regime $A$ $\rightarrow$ "
            r"its rank under our full sweep $B$ (1 $=$ best), ranked among the "
            r"methods shown in this table; Rank$_A$/Rank$_B$ average those over "
            r"the tasks and $\Delta$Rank $=$ Rank$_A-$Rank$_B$ (positive $=$ "
            r"moved up), with " + PROTOCOL_SENTENCE + r". Ranking pool per task "
            r"(methods with a number under both protocols): " + pool_str
            + r". Ties take average ranks. Cells shaded at "
            r"$|\Delta\text{Rank}|\geq0.5$ (darker at $\geq1.5$)."
            + tie_note(tasks, methods, data) + task_legend(tasks))
    lines = [
        r"\begin{table}[t]", r"\centering",
        *([r"\scriptsize"] if len(tasks) > 11 else
          [r"\footnotesize"] if len(tasks) > 8 else []),
        r"\caption{" + caption + "}",
        r"\label{tab:" + SUBSET_TAG[subset] + "-rank-change-"
        + MODE_FILETAG[mode] + lsfx.replace("_", "-") + "}",
        r"\setlength{\tabcolsep}{" + ("2pt" if len(tasks) > 8 else "4pt") + "}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{" + cols + "}", r"\toprule",
        *ghead,
        head + r" \\", r"\midrule",
    ]
    rows = []
    for m in methods:
        cells, ra, rb = [], [], []
        for p in tasks:
            a = ranks[p]["A"].get(m)
            b = ranks[p]["B"].get(m)
            if a is None or b is None:
                cells.append("--")
                continue
            ra.append(a)
            rb.append(b)
            cells.append(f"{a:g}$\\rightarrow${b:g}")
        if not ra:
            continue
        ma, mb = float(np.mean(ra)), float(np.mean(rb))
        rows.append((mb, m, cells, ma, mb))
    for _, m, cells, ma, mb in sorted(rows):          # best final rank first
        d = ma - mb
        lines.append(" & ".join(
            [tex_escape_label(m)] + cells
            + [fmt_rank(ma), fmt_rank(mb),
               f"{shade(d, lo=0.5, hi=1.5)}${d:+.2f}$"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---- Console summary --------------------------------------------------------

def print_summary(tasks, methods, data, ranks, mode):
    print(f"\n===== {MODE_LABEL[mode]} =====")
    print(f"{'method':10s} {'dRR% mean':>10s} {'median':>8s} {'helped':>7s} "
          f"{'rankA':>6s} {'rankB':>6s} {'dRank':>7s}")
    rows = []
    for m in methods:
        vals = []
        for p in tasks:
            r = data.get(p, {}).get(m)
            v = G.rel_gain(r["A"], r["B"]) if r else None
            if v is not None:
                vals.append(v)
        ra = [ranks[p]["A"][m] for p in tasks if m in ranks[p]["A"]]
        rb = [ranks[p]["B"][m] for p in tasks if m in ranks[p]["B"]]
        if not vals or not ra:
            continue
        rows.append((float(np.mean(rb)), m, float(np.mean(vals)),
                     float(np.median(vals)), sum(v > 0 for v in vals), len(vals),
                     float(np.mean(ra)), float(np.mean(rb))))
    for mb, m, mean, med, h, n, ra, rb in sorted(rows):
        print(f"{m:10s} {mean:10.1f} {med:8.1f} {h:4d}/{n:2d} "
              f"{ra:6.2f} {rb:6.2f} {ra - rb:+7.2f}")

    print("\n  levels (A -> B), test relative regret %:")
    for p in tasks:
        cells = []
        for m in methods:
            r = data.get(p, {}).get(m)
            if not r or r["A"] is None or r["B"] is None:
                continue
            cells.append(f"{m}:{r['A']:.3g}->{r['B']:.3g}")
        print(f"    {p:20s} " + "  ".join(cells))


def print_group_summary(tasks, methods, data, subset, mode):
    groups = groups_for(subset, tasks)
    if not groups:
        return
    print(f"\n  per-category dRR% ({MODE_LABEL[mode]}):")
    print("    " + f"{'method':10s}" +
          "".join(f"{lab[:14]:>26s}" for lab, _ in groups))
    print("    " + " " * 10 +
          "".join(f"{'mean':>9s}{'med':>8s}{'helped':>9s}" for _ in groups))
    for m in methods:
        if not dvals(tasks, data, m):
            continue
        row = f"    {m:10s}"
        for _, grp in groups:
            v = dvals(grp, data, m)
            row += (f"{np.mean(v):9.1f}{np.median(v):8.1f}"
                    f"{str(sum(x > 0 for x in v)) + '/' + str(len(v)):>9s}"
                    if v else f"{'--':>9s}{'--':>8s}{'--':>9s}")
        print(row)


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="not_recommended", choices=list(SUBSETS))
    ap.add_argument("--modes", nargs="+", default=["mean", "best", "seed0"],
                    choices=["mean", "best", "seed0"])
    ap.add_argument("--outdir", default="results/tables")
    ap.add_argument("--include_excluded_methods", action="store_true",
                    help=f"keep {sorted(DEFAULT_EXCLUDE)} (dropped by default, "
                         "matching fig_bench_bump_not_recommended.py)")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="METHOD",
                    help="additional methods to drop from the rows AND from the "
                         "ranking pool, e.g. --exclude mse_train mse_val")
    ap.add_argument("--drop_tasks", nargs="*", default=[], metavar="TASK",
                    help="remove tasks from the chosen subset, e.g. "
                         "--drop_tasks shortestpath")
    ap.add_argument("--short_caption", action="store_true",
                    help="caption is one sentence defining the metric only")
    ap.add_argument("--name_suffix", default="",
                    help="appended to output filenames, e.g. _paper")
    ap.add_argument("--use_cache", action="store_true",
                    help=f"reuse {CACHE} instead of rescanning saved_records/")
    args = ap.parse_args()

    tasks = SUBSETS[args.subset]
    bad = [t for t in args.drop_tasks if t not in tasks]
    if bad:
        ap.error(f"--drop_tasks not in the {args.subset} subset: {bad}")
    if args.drop_tasks:
        tasks = [t for t in tasks if t not in set(args.drop_tasks)]
        print(f"[tasks] dropping {sorted(args.drop_tasks)}")
        if not tasks:
            ap.error("--drop_tasks removed every task")
    # Scan set is independent of --exclude so one cached scan serves every
    # display variant; the *displayed* set is what drives rows and ranking.
    excluded = set() if args.include_excluded_methods else DEFAULT_EXCLUDE
    scan_methods = [m for m in G.ALL_METHODS if m not in excluded]
    unknown = [m for m in args.exclude if m not in G.ALL_METHODS]
    if unknown:
        ap.error(f"unknown method(s) in --exclude: {unknown}")
    methods = [m for m in scan_methods if m not in set(args.exclude)]
    if excluded:
        print(f"[methods] excluding {sorted(excluded)} from rows and from the "
              f"ranking pool (--include_excluded_methods to keep)")
    if args.exclude:
        print(f"[methods] also excluding {sorted(args.exclude)} from rows and "
              f"from the ranking pool")
    print(f"[methods] ranking pool = the {len(methods)} displayed methods: "
          f"{methods}")
    print(f"[tasks] {args.subset}: {tasks}")

    # Cache is keyed per (mode, task) so a new task subset only scans the tasks
    # it adds. Older subset-keyed entries are migrated on load.
    cache = {}
    if args.use_cache and os.path.exists(CACHE):
        with open(CACHE) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if k.startswith("t|"):
                cache[k] = v
            else:                                   # legacy "subset|mode|flag"
                parts = k.split("|")
                if len(parts) == 3 and isinstance(v, dict):
                    for prob, rows in v.items():
                        cache.setdefault(f"t|{parts[1]}|{prob}", rows)
        print(f"[cache] loaded {CACHE} ({len(cache)} task-entries)")

    os.makedirs(args.outdir, exist_ok=True)
    written = []
    for mode in args.modes:
        data, scanned = {}, []
        for prob in tasks:
            ckey = f"t|{mode}|{prob}"
            hit = cache.get(ckey)
            if hit and all(m in hit or not G.applies(m, prob) for m in methods):
                data[prob] = hit
            else:
                if not scanned:
                    print(f"[{mode}] scanning saved_records/ ...")
                data[prob] = compute_task(prob, scan_methods, mode)
                cache[ckey] = data[prob]
                scanned.append(prob)
        if scanned:
            with open(CACHE, "w") as f:
                json.dump(cache, f, indent=1)
        print(f"[{mode}] {len(tasks) - len(scanned)} task(s) from cache, "
              f"{len(scanned)} scanned")

        # Restrict to the displayed methods BEFORE ranking, so the ranks shown
        # are ranks among exactly the rows of the table.
        data = {p: {m: r for m, r in data.get(p, {}).items() if m in methods}
                for p in tasks}
        ranks = {p: {"A": rank_map(data[p], "A"), "B": rank_map(data[p], "B")}
                 for p in tasks}

        tag = SUBSET_TAG[args.subset]
        sfx = args.name_suffix
        f1 = os.path.join(args.outdir,
                          f"{tag}_regret_change_{MODE_FILETAG[mode]}{sfx}.tex")
        f2 = os.path.join(args.outdir,
                          f"{tag}_rank_change_{MODE_FILETAG[mode]}{sfx}.tex")
        with open(f1, "w") as f:
            f.write(regret_table(tasks, methods, data, mode, args.subset,
                                 short=args.short_caption, lsfx=sfx))
        with open(f2, "w") as f:
            f.write(rank_table(tasks, methods, data, ranks, mode, args.subset,
                               short=args.short_caption, lsfx=sfx))
        written += [f1, f2]

        gsum = group_summary_table(tasks, methods, data, mode, args.subset,
                                   short=args.short_caption, lsfx=sfx)
        if gsum:
            f3 = os.path.join(
                args.outdir,
                f"{tag}_regret_change_by_group_{MODE_FILETAG[mode]}{sfx}.tex")
            with open(f3, "w") as f:
                f.write(gsum)
            written.append(f3)
            print_group_summary(tasks, methods, data, args.subset, mode)
        print_summary(tasks, methods, data, ranks, mode)

    print("\nwrote:")
    for f in written:
        print(f"  {f}")


if __name__ == "__main__":
    main()
