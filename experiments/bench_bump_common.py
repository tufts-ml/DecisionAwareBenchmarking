"""
bench_bump_common.py
--------------------
Shared config + helpers used by the family of bench-bump plots and the
critical-difference diagram. These helpers load the seed-aggregated JSON
written by ``collect_bench_p2_val.py`` so that the same per-seed data backs
every plot.

Public surface:
  - PROBLEMS, METHODS, METHOD_DISPLAY, PROB_DISPLAY, USE_ABSOLUTE,
    METHOD_PROBLEMS, METHOD_STYLE, jitter_x
  - load_per_seed_data(path) -> nested dict with mean + per-seed values
  - method_style(m), style_ax(...), legend_elements(...)
"""

import json
import os
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ---- Problem config ----
PROBLEMS = [
    "knapsack", "knapsack-real", "energy", "budgetalloc",
    "cubic", "bipartitematching", "portfolio",
    "asurv", "cook_county", "speed_humps",
    "sp_synth", "sp_planted", "pg_misspec", "shortestpath",
]

PROB_DISPLAY = {
    "knapsack":          "Knapsack\n(Gen)",
    "knapsack-real":     "Knapsack\n(Real)",
    "energy":            "Scheduling\n(Energy)",
    "budgetalloc":       "Budget\nAllocation",
    "cubic":             "TopK\n(Cubic)",
    "bipartitematching": "Bipartite\nMatch.",
    "portfolio":         "Portfolio",
    "asurv":             "Whooping Crane\n(TopK)",
    "cook_county":       "Cook Cty.\n(TopK)",
    "speed_humps":       "Speed Humps\n(TopK)",
    "sp_synth":          "Shortest Path\n(5×5)",
    "sp_planted":        "SP Planted\n(5×5)",
    "pg_misspec":        "PG Misspecified",
    "shortestpath":      "Warcraft\n(12×12)",
}

USE_ABSOLUTE = {"portfolio"}

# ---- Method config ----
METHODS = [
    "mse", "mse_train", "mse_val",
    "dfl", "identity", "spo", "nce", "blackbox",
    "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg", "dad",
    "qptl", "cpLayer",
]

METHOD_DISPLAY = {
    "mse":       "MSE",
    "mse_train": "MSE (train-sel)",
    "mse_val":   "MSE (val-sel)",
    "dfl":       "DFL",
    "identity":  "Identity",
    "spo":       "SPO+",
    "nce":       "NCE",
    "blackbox":  "Blackbox",
    "pointLTR":  "pt-LTR",
    "pairLTR":   "pr-LTR",
    "listLTR":   "L-LTR",
    "lodl":      "LODL",
    "perturb":   "DPO",
    "pg":        "PG",
    "dad":       "DAD",
    "qptl":      "QPTL",
    "cpLayer":   "cpLayer",
}

# Methods restricted to a subset of problems (mirrors collect_bench_p1.py)
METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      set(PROBLEMS) - {"shortestpath"},
}

# ---- Marker / color scheme (shared across all bump figures) ----
MSE_C       = "#1f77b4"
MSE_TRAIN_C = "#c39bd3"
MSE_VAL_C   = "#6c3483"
LTRL_C      = "#e67e22"
G1 = "#bbbbbb"; G2 = "#888888"; G3 = "#444444"; G4 = "#222222"
G5 = "#666666"; G6 = "#999999"
DAD_C = "#8e44ad"

METHOD_STYLE = {
    "mse":       ("*",  MSE_C,       80,  1.0),
    "mse_train": ("*",  MSE_TRAIN_C, 80,  1.0),
    "mse_val":   ("*",  MSE_VAL_C,   80,  1.0),
    "dfl":       ("h",  G1,     50,  0.90),
    "blackbox":  ("h",  G2,     50,  0.90),
    "identity":  ("h",  G3,     50,  0.90),
    "perturb":   ("h",  G4,     50,  0.90),
    "spo":       ("P",  G5,     50,  0.90),
    "pg":        ("P",  G6,     50,  0.90),
    "lodl":      ("o",  G2,     44,  0.85),
    "dad":       ("D",  DAD_C,  42,  0.90),
    "cpLayer":   ("^",  G1,     50,  0.90),
    "qptl":      ("^",  G2,     50,  0.90),
    "nce":       ("s",  G1,     44,  0.85),
    "pointLTR":  ("s",  G2,     44,  0.85),
    "pairLTR":   ("s",  G3,     44,  0.85),
    "listLTR":   ("s",  LTRL_C, 50,  0.90),
}

def method_style(m):
    return METHOD_STYLE.get(m, ("o", G2, 38, 0.7))


# Same horizontal jitter pattern as the original bump, so all variants line up.
np.random.seed(42)
jitter_x = {m: float(np.random.uniform(-0.28, 0.28)) for m in METHODS}
jitter_x["mse"]       = -0.30
jitter_x["mse_train"] = -0.15
jitter_x["mse_val"]   = -0.22
jitter_x["listLTR"]   =  0.0
jitter_x["perturb"]   =  0.0
jitter_x["dad"]       =  0.18


# ============================================================
# Data loading
# ============================================================

def load_per_seed_data(path="bench_p2_best_val.json"):
    """
    Returns dict: out[method][prob] = {
        "mean":     scalar in display units (relative-% or absolute),
        "seeds":    list[float] per-seed values in display units,
        "n_done":   int,
        "n_total":  int,
        "absolute": bool,    # True for portfolio, False otherwise
    }

    Missing cells are absent from the inner dict. ``mean``/``seeds`` are
    already scaled to display units (×100 for relative-regret problems,
    unscaled for portfolio).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run "
            "`python experiments/collect_bench_p2_val.py` first.")
    with open(path) as f:
        raw = json.load(f)

    out = {}
    for method in METHODS:
        out[method] = {}
        cells = raw.get(method, {})
        for prob in PROBLEMS:
            allowed = METHOD_PROBLEMS.get(method)
            if allowed is not None and prob not in allowed:
                continue
            entry = cells.get(prob)
            if entry is None:
                continue
            seeds_dict = entry.get("test_seed_values") or {}
            if not seeds_dict:
                continue
            absolute = prob in USE_ABSOLUTE
            scale = 1.0 if absolute else 100.0
            # Order by seed id for determinism.
            ordered_keys = sorted(seeds_dict.keys(), key=lambda s: int(s))
            seeds = [float(seeds_dict[k]) * scale for k in ordered_keys]
            by_seed = {int(k): float(seeds_dict[k]) * scale for k in ordered_keys}
            mean_v = entry.get("test")
            if mean_v is None:
                mean_v = float(np.mean(seeds))
            else:
                mean_v = float(mean_v) * scale
            out[method][prob] = {
                "mean":    mean_v,
                "seeds":   seeds,
                "by_seed": by_seed,
                "n_done":  int(entry.get("test_n_done") or len(seeds)),
                "n_total": int(entry.get("n_total") or len(seeds)),
                "absolute": absolute,
            }
    return out


def get_means(data):
    """Convenience: dict[method][prob] -> mean (NaN for missing)."""
    out = {}
    for m in METHODS:
        out[m] = {}
        for p in PROBLEMS:
            cell = data.get(m, {}).get(p)
            out[m][p] = float(cell["mean"]) if cell else float("nan")
    return out


# ============================================================
# Axis styling (shared)
# ============================================================

def _choose_5_ticks(v_min, v_max):
    lo = np.log10(max(v_min, 1e-9))
    hi = np.log10(max(v_max, lo + 1e-6))
    return 10 ** np.linspace(lo, hi, 5)


def _fmt_val(x, is_portfolio):
    # 2 decimals for x < 1, 1 decimal for x < 100, 0 decimals otherwise.
    if x < 1.0:
        s = f"{x:.2f}"
    elif x < 100.0:
        s = f"{x:.1f}"
    else:
        s = f"{x:.0f}"
    return s if is_portfolio else s + "%"


def style_ax(ax, prob, values_for_extent, show_ylabel=False, pad_frac=0.15):
    """
    Style a single bump-chart axis (log-y, inverted, 5 ticks, etc.). The y
    extent is computed from ``values_for_extent`` — pass the full set of
    values (including any error-bar endpoints) so error bars don't get
    clipped.
    """
    is_portfolio = prob in USE_ABSOLUTE
    vals = [v for v in values_for_extent if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        return
    v_lo, v_hi = min(vals), max(vals)
    log_range = np.log10(v_hi) - np.log10(max(v_lo, 1e-9))
    log_pad = log_range * pad_frac + 0.005
    v_min = max(10 ** (np.log10(max(v_lo, 1e-9)) - log_pad), 1e-4)
    v_max = 10 ** (np.log10(v_hi) + log_pad)

    ticks = _choose_5_ticks(v_min, v_max)
    ax.set_yscale("log")
    ax.set_ylim(v_max, v_min)
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([])

    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks(ticks)
    fmt_fn = lambda x, _: _fmt_val(x, is_portfolio)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_fn))
    ax.tick_params(axis="y", labelsize=6.5, length=0, pad=-3)

    if show_ylabel:
        unit = "Regret (log scale)" if is_portfolio else "Relative regret (log scale)"
        ax.set_ylabel(unit, fontsize=7, fontweight="bold", labelpad=24)

    for y in ticks:
        ax.axhline(y, color="#dddddd", linewidth=0.5, zorder=0)

    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["left"].set_zorder(1)


def rewrite_ticklabels_with_boxes(fig, ax):
    """Re-render y-axis tick labels as white-boxed text on the plot area
    (matches the look used by the original bump chart)."""
    fig.canvas.draw()
    tp = ax.get_yticks()
    tl = [t.get_text() for t in ax.get_yticklabels()]
    ax.set_yticklabels([""] * len(tp))
    for yv, txt in zip(tp, tl):
        if not txt:
            continue
        ax.text(0.0, yv, txt, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="square,pad=0.15"),
                zorder=10, clip_on=False)


# ============================================================
# Legend (shared, with exclusions)
# ============================================================

def legend_elements(exclude_methods=()):
    def _le(mkr, col, lbl, ms=7):
        return Line2D([0], [0], marker=mkr, color="w",
                      markerfacecolor=col, markeredgecolor="white",
                      markersize=ms, markeredgewidth=0.5, label=lbl)
    def _hdr(t):
        return Line2D([0], [0], color="none",
                      label=r"$\bf{" + t.replace(" ", r"\ ") + r"}$")
    def _star(col, lbl):
        return Line2D([0], [0], marker="*", color="w", markerfacecolor=col,
                      markeredgecolor="white", markersize=10, markeredgewidth=0.8,
                      label=lbl)
    elements = [
        _hdr("Decision-Unaware (MSE)"),
        _star(MSE_C,       "MSE  (val regret)"),
        _star(MSE_VAL_C,   "MSE  (val MSE)"),
        _star(MSE_TRAIN_C, "MSE  (train MSE)"),
        _hdr("Surrogate Gradient"),
        _le("h", G1, "DFL"), _le("h", G2, "Blackbox"),
        _le("h", G3, "Identity"), _le("h", G4, "DPO"),
        _le("P", G5, "SPO+"), _le("P", G6, "PG"),
        _hdr("Trained Surrogate"),
        _le("o", G2, "LODL"),
    ]
    if "dad" not in exclude_methods:
        elements += [
            _hdr("Decision-Aware Denoising"),
            _le("D", DAD_C, "DAD"),
        ]
    cont_pairs = [("cpLayer", _le("^", G1, "cpLayer")),
                  ("qptl",    _le("^", G2, "QPTL"))]
    cont_kept = [le for (m, le) in cont_pairs if m not in exclude_methods]
    if cont_kept:
        elements.append(_hdr("Continuous"))
        elements.extend(cont_kept)
    elements += [
        _hdr("Statistical"),
        _le("s", G1, "NCE"),    _le("s", G2, "pt-LTR"),
        _le("s", G3, "pr-LTR"), _le("s", LTRL_C, "L-LTR"),
    ]
    return elements


# ============================================================
# Chart groupings (same as original bump)
# ============================================================
CLASSIC   = ["knapsack", "knapsack-real", "energy", "cubic", "bipartitematching"]
SPATIAL   = ["asurv", "cook_county", "speed_humps"]
SP_FAMILY = ["sp_synth", "sp_planted", "shortestpath"]


# ============================================================
# Critical-difference helpers (shared by cross-task and per-task CD)
# ============================================================

# Demšar 2006, Table 5: Nemenyi q_alpha = studentized range / sqrt(2).
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489,
    19: 3.517, 20: 3.544,
}
NEMENYI_Q10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
    8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077,
    14: 3.120, 15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261,
    19: 3.291, 20: 3.319,
}


def nemenyi_cd(k, N, alpha=0.05):
    """Nemenyi critical difference: CD = q_α · sqrt(k(k+1) / 6N)."""
    table = NEMENYI_Q05 if alpha == 0.05 else NEMENYI_Q10
    if k not in table:
        raise ValueError(f"No tabled q value for k={k}, alpha={alpha}")
    q = table[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * N))


def rank_within_rows(mat):
    """Rank each row of ``mat`` ascending (1 = smallest). Ties get avg rank."""
    from scipy.stats import rankdata
    return np.array([rankdata(row, method="average") for row in mat])


def draw_cd_diagram(avg_ranks, methods, cd, title, out_path,
                    friedman_p=None, alpha=0.05, figsize=None,
                    pdfpages=None):
    """
    Render a Demšar/Nemenyi CD diagram from a precomputed (avg_ranks, cd)
    pair.

    Parameters
    ----------
    avg_ranks : sequence of float
        Average ranks per method, same order as ``methods``.
    methods : sequence of str
        Method identifiers (used to look up display name + marker style).
    cd : float
        Critical difference width on the rank axis.
    title : str
        Figure title.
    out_path : str
        Output PNG path. A sibling PDF is also written.
    friedman_p : float or None
        If provided, included in the title.
    alpha : float
        Significance level (used only for the title annotation).
    figsize : (w, h) or None
        Auto-sized to fit all label rows when None.

    Returns
    -------
    dict with avg ranks per method, cd, cliques, and friedman_p.
    """
    import matplotlib.pyplot as plt

    k = len(methods)
    avg_ranks = list(avg_ranks)

    # ---- Plot geometry ----
    n_right = (k + 1) // 2
    n_left  = k - n_right
    row_spacing = 0.16
    label_top_y = -0.30
    rows_per_side = max(n_right, n_left)
    label_bottom_y = label_top_y - (rows_per_side - 1) * row_spacing
    y_lo = label_bottom_y - 0.20
    y_hi = 1.0
    if figsize is None:
        fig_h = 2.0 + 0.36 * rows_per_side
        figsize = (11.0, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(k + 0.5, 0.5)   # invert: best (rank 1) on right
    ax.set_ylim(y_lo, y_hi)

    # Rank axis
    ax.plot([0.5, k + 0.5], [0, 0], color="black", linewidth=1.0)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.05], color="black", linewidth=0.8)
        ax.text(r, 0.10, str(r), ha="center", va="bottom", fontsize=8)
    ax.text((k + 1) / 2.0, 0.32, "Average rank  (best ↑)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # CD bar in the upper-right (just past rank 1)
    cd_y = 0.55
    cd_left  = 1.0
    cd_right = 1.0 + cd
    ax.plot([cd_left, cd_right], [cd_y, cd_y], color="black", linewidth=1.2)
    ax.plot([cd_left,  cd_left],  [cd_y - 0.04, cd_y + 0.04], color="black", linewidth=1.2)
    ax.plot([cd_right, cd_right], [cd_y - 0.04, cd_y + 0.04], color="black", linewidth=1.2)
    ax.text((cd_left + cd_right) / 2.0, cd_y + 0.08,
            f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)

    # Labels: best half on right, worst half on left
    pairs = sorted(zip(methods, avg_ranks), key=lambda x: x[1])
    right = pairs[:n_right]
    left  = pairs[n_right:]
    label_x_right = 0.6
    label_x_left  = k + 0.4
    for i, (m, r) in enumerate(right):
        y_row = label_top_y - i * row_spacing
        mkr, col, sz, _ = method_style(m)
        ax.plot([r, r], [0, y_row], color="black", linewidth=0.6)
        ax.plot([r, label_x_right], [y_row, y_row], color="black", linewidth=0.6)
        ax.scatter([label_x_right - 0.05], [y_row], s=sz * 0.75, marker=mkr,
                   color=col, edgecolors="white", linewidths=0.5, zorder=4)
        ax.text(label_x_right - 0.18, y_row,
                f"{METHOD_DISPLAY.get(m, m)}  ({r:.2f})",
                ha="left", va="center", fontsize=8)
    for i, (m, r) in enumerate(left):
        y_row = label_top_y - i * row_spacing
        mkr, col, sz, _ = method_style(m)
        ax.plot([r, r], [0, y_row], color="black", linewidth=0.6)
        ax.plot([r, label_x_left], [y_row, y_row], color="black", linewidth=0.6)
        ax.scatter([label_x_left + 0.05], [y_row], s=sz * 0.75, marker=mkr,
                   color=col, edgecolors="white", linewidths=0.5, zorder=4)
        ax.text(label_x_left + 0.18, y_row,
                f"({r:.2f})  {METHOD_DISPLAY.get(m, m)}",
                ha="right", va="center", fontsize=8)

    # Cliques: maximal contiguous groups within CD of each other.
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and pairs[j + 1][1] - pairs[i][1] < cd:
            j += 1
        if j > i:
            cliques.append((pairs[i][1], pairs[j][1]))
        i += 1

    def _contained(a, b):
        return a[0] >= b[0] and a[1] <= b[1] and a != b
    cliques = [c for c in cliques if not any(_contained(c, d) for d in cliques)]

    clique_y0 = -0.08
    for idx, (lo, hi) in enumerate(cliques):
        y = clique_y0 - idx * 0.05
        ax.plot([lo - 0.05, hi + 0.05], [y, y],
                color="black", linewidth=3.0, solid_capstyle="butt")

    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    full_title = title
    if friedman_p is not None:
        full_title = f"{title}\nFriedman p = {friedman_p:.2e}"
    fig.suptitle(full_title, fontsize=10, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.04, right=0.96)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path[:-4] + ".pdf" if out_path.endswith(".png") else out_path + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    if pdfpages is not None:
        pdfpages.savefig(fig, bbox_inches="tight")
    plt.close()

    return {
        "avg_ranks": dict(zip(methods, [float(r) for r in avg_ranks])),
        "cd": float(cd),
        "cliques": [(float(lo), float(hi)) for lo, hi in cliques],
        "friedman_p": None if friedman_p is None else float(friedman_p),
    }
