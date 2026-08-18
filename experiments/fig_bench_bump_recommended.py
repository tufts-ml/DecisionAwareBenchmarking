"""
fig_bench_bump_recommended.py
-----------------------------
Bump charts for the recommended task subset, split into two groups
(each with an average-rank column):

  Group "main"    : budget allocation, PG misspec, shortest path (5x5)
  Group "spatial" : cook county (TopK), speed humps (TopK), asurv (TopK)

Excludes DAD, QPTL, and cpLayer from methods.

Generates four figures (two groups x two uncertainty modes):
  1. Plain 10-seed mean (no uncertainty)
  2. 10-seed mean +/- 95% bootstrap CI

Usage::

    python rethink_exp/fig_bench_bump_recommended.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from bench_bump_common import (
    METHODS, METHOD_DISPLAY, PROB_DISPLAY,
    USE_ABSOLUTE, METHOD_PROBLEMS, jitter_x,
    load_per_seed_data, method_style, style_ax, rewrite_ticklabels_with_boxes,
    legend_elements,
)

TASK_GROUPS = {
    "main":    ["budgetalloc", "pg_misspec", "sp_synth"],
    "spatial": ["cook_county", "speed_humps", "asurv"],
}

EXCLUDE = {"dad", "qptl", "cpLayer"}
ACTIVE_METHODS = [m for m in METHODS if m not in EXCLUDE]

N_RESAMPLES = 10_000
CI_LEVEL = 0.95
RNG_SEED = 20260527

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

data = load_per_seed_data("bench_p2_best_val.json")

try:
    from scipy.stats import bootstrap as _scipy_bootstrap
    _HAS_SCIPY_BS = True
except Exception:
    _HAS_SCIPY_BS = False


def bootstrap_ci_mean(seeds, n_resamples=N_RESAMPLES, level=CI_LEVEL,
                      rng_seed=RNG_SEED):
    seeds = np.asarray(seeds, dtype=float)
    if len(seeds) < 2:
        return float(np.mean(seeds)), float(np.mean(seeds))
    if _HAS_SCIPY_BS:
        try:
            res = _scipy_bootstrap(
                (seeds,), np.mean, n_resamples=n_resamples,
                confidence_level=level, method="BCa",
                random_state=np.random.default_rng(rng_seed),
            )
            return float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            pass
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)))
    means = seeds[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _cell_summary(method, prob, with_ci=False):
    cell = data.get(method, {}).get(prob)
    if cell is None or not cell["seeds"]:
        return None
    mean = cell["mean"]
    n_done = cell["n_done"]
    if with_ci and n_done > 1:
        lo, hi = bootstrap_ci_mean(cell["seeds"])
    else:
        lo, hi = mean, mean
    return mean, lo, hi, n_done


def _column_value_set(prob, with_ci=False):
    vals = []
    for m in ACTIVE_METHODS:
        allowed = METHOD_PROBLEMS.get(m)
        if allowed is not None and prob not in allowed:
            continue
        s = _cell_summary(m, prob, with_ci=with_ci)
        if s is None:
            continue
        mean, lo, hi, _ = s
        vals.extend([mean, lo, hi])
    return vals


plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.bottom": False,
})


def draw(tasks, group, with_ci=False):
    n_tasks = len(tasks)
    n_cols = n_tasks + 1
    width_ratios = [1] * n_tasks + [1.1]
    figsize = (max(3.0, 1.4 * n_tasks + 2.6), 3.2)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, n_cols, figure=fig,
                  width_ratios=width_ratios, wspace=0.10)
    axes = [fig.add_subplot(gs[0, j]) for j in range(n_tasks)]

    for j, (ax, prob) in enumerate(zip(axes, tasks)):
        style_ax(ax, prob, _column_value_set(prob, with_ci=with_ci),
                 show_ylabel=(j == 0))
        for m in ACTIVE_METHODS:
            allowed = METHOD_PROBLEMS.get(m)
            if allowed is not None and prob not in allowed:
                continue
            s = _cell_summary(m, prob, with_ci=with_ci)
            if s is None:
                continue
            mean, lo, hi, n_done = s
            mkr, col, sz, alp = method_style(m)
            x = jitter_x[m]
            if with_ci and n_done > 1 and hi > lo:
                ax.plot([x, x], [lo, hi], color=col, linewidth=1.4,
                        alpha=min(1.0, alp), zorder=2,
                        solid_capstyle="round")
            ax.scatter(x, mean, s=sz, color=col,
                       edgecolors="white", linewidths=0.5,
                       zorder=3, marker=mkr, alpha=alp)
        ax.set_xlabel(PROB_DISPLAY.get(prob, prob), fontsize=6.5,
                      fontweight="bold", labelpad=3)

    # ---- Avg rank column ----
    ax_avg = fig.add_subplot(gs[0, n_tasks])
    avg_rank = {}
    for m in ACTIVE_METHODS:
        ranks = []
        for prob in tasks:
            allowed = METHOD_PROBLEMS.get(m)
            if allowed is not None and prob not in allowed:
                continue
            col_vals = {}
            for mm in ACTIVE_METHODS:
                a2 = METHOD_PROBLEMS.get(mm)
                if a2 is not None and prob not in a2:
                    continue
                s = _cell_summary(mm, prob)
                if s is not None:
                    col_vals[mm] = s[0]
            if m not in col_vals:
                continue
            srt = sorted(col_vals, key=col_vals.__getitem__)
            ranks.append(srt.index(m) + 1)
        avg_rank[m] = float(np.mean(ranks)) if ranks else np.nan

    valid = [v for v in avg_rank.values() if np.isfinite(v)]
    max_rank = max(valid) if valid else len(ACTIVE_METHODS)

    ax_avg.set_ylim(max_rank + 0.5, 0.5)
    ax_avg.set_xlim(-0.45, 0.45)
    ax_avg.set_xticks([])
    labeled = [r for r in range(1, int(max_rank) + 1) if r % 2 == 1]
    ax_avg.set_yticks(labeled)
    ax_avg.set_yticklabels([str(r) for r in labeled])
    ax_avg.tick_params(axis="y", labelsize=6.5, length=0, pad=2)
    ax_avg.spines["left"].set_linewidth(0.6)
    ax_avg.spines["left"].set_color("#cccccc")
    for sp in ("top", "right", "bottom"):
        ax_avg.spines[sp].set_visible(False)
    for r in range(1, int(max_rank) + 1):
        ax_avg.axhline(r, color="#dddddd", linewidth=0.5, zorder=0)
    for m in ACTIVE_METHODS:
        ar = avg_rank.get(m, np.nan)
        if not np.isfinite(ar):
            continue
        mkr, col, sz, alp = method_style(m)
        ax_avg.scatter(jitter_x[m], ar, s=sz, color=col,
                       edgecolors="white", linewidths=0.5,
                       zorder=3, marker=mkr, alpha=alp)
    ax_avg.set_xlabel("Avg\nRank", fontsize=6.5, fontweight="bold", labelpad=3)

    # ---- Legend ----
    legend_els = legend_elements(exclude_methods=EXCLUDE)
    fig.legend(handles=legend_els, loc="upper left",
               bbox_to_anchor=(0.82, 0.94), ncol=1,
               frameon=True, framealpha=0.88, edgecolor="#cccccc",
               fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)
    plt.subplots_adjust(top=0.88, bottom=0.22, left=0.08, right=0.81)

    if with_ci:
        ci_pct = int(CI_LEVEL * 100)
        tag = "bootstrap"
        title = (f"Benchmark re-run — 10-seed mean ± {ci_pct}% bootstrap CI "
                 f"— Relative regret % (best ↑)")
    else:
        tag = "plain"
        title = ("Benchmark re-run — 10-seed mean "
                 "— Relative regret % (best ↑)")

    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.01)

    all_axes = axes + [ax_avg]
    for ax in all_axes:
        ax.spines["left"].set_zorder(2)
        rewrite_ticklabels_with_boxes(fig, ax)

    fname = f"fig_bench_bump_recommended_{group}_{tag}"
    out_png = os.path.join(OUT_DIR, f"{fname}.png")
    out_pdf = os.path.join(OUT_DIR, f"{fname}.pdf")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}  +  {out_pdf}")

    print(f"\n{'Method':<16}", end="")
    for p in tasks:
        print(f"  {p:>14}", end="")
    print(f"  {'Avg Rank':>9}")
    for m in ACTIVE_METHODS:
        print(f"{METHOD_DISPLAY.get(m, m):<16}", end="")
        for p in tasks:
            allowed = METHOD_PROBLEMS.get(m)
            if allowed is not None and p not in allowed:
                print(f"  {'N/A':>14}", end="")
                continue
            s = _cell_summary(m, p)
            if s is None:
                print(f"  {'--':>14}", end="")
            else:
                print(f"  {s[0]:>14.4f}", end="")
        ar = avg_rank.get(m, np.nan)
        if np.isfinite(ar):
            print(f"  {ar:>9.2f}")
        else:
            print(f"  {'--':>9}")


for _group, _tasks in TASK_GROUPS.items():
    draw(_tasks, _group, with_ci=False)
    draw(_tasks, _group, with_ci=True)
