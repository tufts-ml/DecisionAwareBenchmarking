"""
fig_synthetic_data.py
---------------------
Illustration of the pg_misspec synthetic data-generating process
(Gupta & Huang 2024, arXiv 2402.03256, Sec. 4.1 variant 3), as shown in the
paper appendix "PG Misspecified Synthetic Task":

    x  ~ Uniform(0, 2)
    f*(x) = -4x + 2   for x in [0, 0.55],   -0.2  for x in (0.55, 2]
    Y  = f*(x) + zeta - 0.5,   zeta ~ Exponential(mean 0.5)

This is exactly the DGP in openpto/problems/PGMisspec.py — the benchmark
that produced every pg_misspec number in the paper — and the corrected
manuscript equation: eps = zeta - 0.5 gives a hard floor at f*(x) - 0.5
with sparse upward spikes (unbounded upper tail). The RNG recipe (seed,
draw order) is unchanged from the originally submitted figure, so these
samples are the exact mirror of the previously published points about f*.

Generates:
    results/figures/synthetic_data.pdf
    results/figures/synthetic_data.png

Usage::

    python experiments/fig_synthetic_data.py
"""
import sys

if len(sys.argv) > 1:
    print(__doc__ or "Regenerates paper figures into results/figures/; "
          "takes no arguments. Run from the repo root.")
    sys.exit(0 if {"-h", "--help"} & set(sys.argv[1:]) else 2)


import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# DGP parameters (PGMisspec defaults: m=0, |m0|=4, |c0|=0.2, sd=0.5, alpha=1)
M = 0.0      # slope after the kink
M0 = 4.0     # slope magnitude before the kink
C0 = 0.2     # |cost| after the kink
SD = 0.5     # exponential noise mean / scale
ALPHA = 1.0  # noise mixture weight (1 = pure exponential)
T_KINK = (2.0 + C0) / M0  # 0.55, where -M0*x + 2 meets -C0

N_SAMPLES = 1000
SEED = 999  # seed used for the published figure

OUT_DIR = os.path.join("results", "figures")


def f_star(x):
    """True mean cost: -4x + 2 on [0, 0.55], -0.2 on (0.55, 2]."""
    return np.where(x <= T_KINK, -M0 * x + 2.0, -M * (x - T_KINK) - C0)


def gen_data(n, seed):
    """Sample (x, Y) from the pg_misspec DGP (benchmark noise sign).

    The Gaussian draw has weight 0 at ALPHA=1 but must stay in this order
    (uniform, normal, exponential) to reproduce the published figure's
    RNG stream exactly.
    """
    np.random.seed(seed)
    x = np.random.uniform(0.0, 2.0, n)
    norm_noise = np.random.normal(0.0, SD, n)
    exp_noise = np.random.exponential(SD, n) - SD  # zeta - 0.5, mean zero
    noise = exp_noise * ALPHA**0.5 + norm_noise * (1.0 - ALPHA) ** 0.5
    return x, f_star(x) + noise


def main():
    x, y = gen_data(N_SAMPLES, SEED)
    x_dense = np.linspace(0.0, 2.0, 500)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(x_dense, f_star(x_dense), "b-", linewidth=2, label=r"$f^*(x)$",
            zorder=10)
    ax.scatter(x, y, alpha=0.5, s=30, c="orange", label="Y")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("x (feature)", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_title("Synthetic data", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"synthetic_data.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
