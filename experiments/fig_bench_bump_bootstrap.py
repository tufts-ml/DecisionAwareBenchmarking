"""
fig_bench_bump_bootstrap.py
---------------------------
Bench-bump charts with bootstrap 95% confidence intervals on the mean.

Each method's marker sits at the mean test regret across the available
init seeds; a thin vertical bar shows a bootstrap percentile 95% CI of the
mean (10000 resamples; BCa via scipy when available, percentile method
otherwise). Cells with one seed get no CI.

Reads ``bench_p2_best_val.json`` (which carries ``test_seed_values``).

Usage::

    python experiments/fig_bench_bump_bootstrap.py

Tweak ``N_RESAMPLES`` and ``CI_LEVEL`` at the top of the file.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bench_bump_common import (
    PROBLEMS, METHODS, METHOD_DISPLAY, PROB_DISPLAY,
    USE_ABSOLUTE, METHOD_PROBLEMS, METHOD_STYLE, jitter_x,
    load_per_seed_data, method_style, style_ax, rewrite_ticklabels_with_boxes,
    legend_elements, CLASSIC, SPATIAL, SP_FAMILY,
)

# ---- Bootstrap config ----
N_RESAMPLES = 10000
CI_LEVEL    = 0.95
RNG_SEED    = 20260524     # deterministic CIs

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

data = load_per_seed_data("bench_p2_best_val.json")


# ---- Optional: BCa via scipy if available ----
try:
    from scipy.stats import bootstrap as _scipy_bootstrap
    _HAS_SCIPY_BS = True
except Exception:
    _HAS_SCIPY_BS = False


def bootstrap_ci_mean(seeds, n_resamples=N_RESAMPLES, level=CI_LEVEL,
                      rng_seed=RNG_SEED):
    """Return (lo, hi) bootstrap CI for the mean of ``seeds``.

    Prefers scipy's BCa method (bias-corrected and accelerated) since the
    percentile method is slightly anti-conservative at small N. Falls back
    to the percentile method if scipy isn't available.
    """
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
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi


plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.bottom": False,
})


def _cell_summary(method, prob):
    """Return (mean, ci_lo, ci_hi, n_done) or None if missing."""
    cell = data.get(method, {}).get(prob)
    if cell is None or not cell["seeds"]:
        return None
    seeds = cell["seeds"]
    mean = cell["mean"]
    n_done = cell["n_done"]
    if n_done > 1:
        lo, hi = bootstrap_ci_mean(seeds)
    else:
        lo, hi = mean, mean
    return mean, lo, hi, n_done


def _column_value_set(prob, methods):
    vals = []
    for m in methods:
        s = _cell_summary(m, prob)
        if s is None:
            continue
        mean, lo, hi, _ = s
        vals.extend([mean, lo, hi])
    return vals


def draw(task_list, title, fname, figsize=None, add_avg_rank=False,
         exclude_methods=()):
    n_tasks = len(task_list)
    if figsize is None:
        figsize = (max(3.0, 1.4 * n_tasks + 2.6), 3.2)

    single = (n_tasks == 1)
    n_cols = n_tasks + (1 if add_avg_rank else 0)
    width_ratios = [1] * n_tasks + ([1.1] if add_avg_rank else [])

    methods = [m for m in METHODS if m not in exclude_methods]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, n_cols, figure=fig,
                  width_ratios=width_ratios, wspace=0.10)
    axes = [fig.add_subplot(gs[0, j]) for j in range(n_tasks)]

    for j, (ax, prob) in enumerate(zip(axes, task_list)):
        style_ax(ax, prob, _column_value_set(prob, methods), show_ylabel=(j == 0))
        for m in methods:
            s = _cell_summary(m, prob)
            if s is None:
                continue
            mean, lo, hi, n_done = s
            mkr, col, sz, alp = method_style(m)
            x = jitter_x[m]
            if n_done > 1 and hi > lo:
                ax.plot([x, x], [lo, hi], color=col, linewidth=1.4,
                        alpha=min(1.0, alp), zorder=2,
                        solid_capstyle="round")
            ax.scatter(x, mean, s=sz, color=col,
                       edgecolors="white", linewidths=0.5,
                       zorder=3, marker=mkr, alpha=alp)
        ax.set_xlabel(PROB_DISPLAY.get(prob, prob), fontsize=6.5,
                      fontweight="bold", labelpad=3)

    ax_avg = None
    if add_avg_rank:
        ax_avg = fig.add_subplot(gs[0, n_tasks])
        avg_rank = {}
        for m in methods:
            ranks = []
            for prob in task_list:
                col_vals = {}
                for mm in methods:
                    s = _cell_summary(mm, prob)
                    if s is not None:
                        col_vals[mm] = s[0]
                if m not in col_vals:
                    continue
                srt = sorted(col_vals, key=col_vals.__getitem__)
                ranks.append(srt.index(m) + 1)
            avg_rank[m] = float(np.mean(ranks)) if ranks else np.nan
        valid = [v for v in avg_rank.values() if np.isfinite(v)]
        max_rank = max(valid) if valid else len(methods)

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
        for m in methods:
            ar = avg_rank.get(m, np.nan)
            if not np.isfinite(ar):
                continue
            mkr, col, sz, alp = method_style(m)
            ax_avg.scatter(jitter_x[m], ar, s=sz, color=col,
                           edgecolors="white", linewidths=0.5,
                           zorder=3, marker=mkr, alpha=alp)
        ax_avg.set_xlabel("Avg\nRank", fontsize=6.5, fontweight="bold", labelpad=3)

    legend_els = legend_elements(exclude_methods=exclude_methods)
    if single:
        fig.legend(handles=legend_els, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncol=2,
                   frameon=True, framealpha=0.88, edgecolor="#cccccc",
                   fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)
        plt.subplots_adjust(top=0.85, bottom=0.03, left=0.20, right=0.95)
    else:
        fig.legend(handles=legend_els, loc="upper left",
                   bbox_to_anchor=(0.82, 0.94), ncol=1,
                   frameon=True, framealpha=0.88, edgecolor="#cccccc",
                   fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)
        plt.subplots_adjust(top=0.88, bottom=0.22, left=0.08, right=0.81)

    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.01)

    all_axes = axes + ([ax_avg] if ax_avg is not None else [])
    for ax in all_axes:
        ax.spines["left"].set_zorder(2)
        rewrite_ticklabels_with_boxes(fig, ax)

    out_png = os.path.join(OUT_DIR, fname)
    out_pdf = os.path.join(OUT_DIR, fname.replace(".png", ".pdf"))
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}  +  {out_pdf}")


ci_label = f"{int(CI_LEVEL * 100)}% bootstrap CI of mean"

draw(
    PROBLEMS,
    title=f"Benchmark re-run — All {len(PROBLEMS)} tasks — mean ± {ci_label} (n≤10 seeds) — Relative regret % (best ↑)",
    fname="fig_bench_bump_bootstrap_all.png",
    figsize=(15.5, 3.4),
    add_avg_rank=True,
)

draw(
    CLASSIC,
    title=f"Benchmark re-run — Classic tasks — mean ± {ci_label} — Relative regret % (best ↑)",
    fname="fig_bench_bump_bootstrap_classic.png",
    figsize=(8.5, 3.2),
    add_avg_rank=True,
    exclude_methods=["dad", "cpLayer", "qptl"],
)

draw(
    SPATIAL,
    title=f"Benchmark re-run — Spatial TopK tasks — mean ± {ci_label}",
    fname="fig_bench_bump_bootstrap_spatial.png",
    figsize=(6.0, 3.2),
    add_avg_rank=True,
)

draw(
    SP_FAMILY,
    title=f"Benchmark re-run — Shortest-path tasks — mean ± {ci_label}",
    fname="fig_bench_bump_bootstrap_sp.png",
    figsize=(6.0, 3.2),
    add_avg_rank=True,
)

draw(
    ["budgetalloc"],
    title=f"Budget Allocation\nmean ± {ci_label} — Relative regret % (best ↑)",
    fname="fig_bench_bump_bootstrap_budgetalloc.png",
    figsize=(2.2, 2.8),
    exclude_methods=["dad", "cpLayer", "qptl"],
)

draw(
    ["portfolio"],
    title=f"Portfolio\nmean ± {ci_label} — Absolute regret (best ↑)",
    fname="fig_bench_bump_bootstrap_portfolio.png",
    figsize=(2.2, 2.8),
    exclude_methods=["dad", "cpLayer", "qptl"],
)
