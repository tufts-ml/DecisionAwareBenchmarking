"""
sweep_status.py
---------------
Status dashboard for the benchmark re-run sweep (Phase 1 and Phase 2).

Reads a manifest JSON produced by submit_bench_p1.sh or submit_bench_p2.sh and
checks the on-disk state of each (problem, opt_model, prefix) tuple.

Status codes:
  ✓  = results.npy exists (complete)
  R  = checkpoint_latest.pt exists but no results.npy (in-progress / preempted)
  -- = nothing (not started or failed before first checkpoint)

Usage:
    python rethink_exp/sweep_status.py --phase 1
    python rethink_exp/sweep_status.py --phase 1 --vals
    python rethink_exp/sweep_status.py --manifest sweep_manifest_p1.json
    python rethink_exp/sweep_status.py --manifest sweep_manifest_p2.json --vals
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

RESULTS_ROOT = "saved_records"


def get_queued_job_names():
    """Return set of SLURM job names currently queued for the user."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return set(out.strip().split("\n")) - {""}
    except Exception:
        return set()


def prefix_to_job_name(phase, prob, method, prefix):
    """Map prefix → SLURM job name set by the submit scripts.

    Phase 1: bp1_<prob>_<method>_<batch>_lr<lr>     (prefix == bench_p1_<method>_<batch>_lr<lr>)
    Phase 2: bp2_<prob>_<method>_<hp_tag>            (prefix == bench_p2_<method>_<hp_tag>_<batch>_lr<lr>)
    """
    if phase == 1:
        # bench_p1_<method>_<rest> → bp1_<prob>_<method>_<rest>
        rest = prefix[len(f"bench_p1_{method}_"):]
        return f"bp1_{prob}_{method}_{rest}"
    # Phase 2: strip the trailing "_<batch>_lr<lr>" from prefix to get hp_tag
    import re
    rest = prefix[len(f"bench_p2_{method}_"):]
    m = re.match(r"(.+)_(default|alt)_lr([^_]+)$", rest)
    hp_tag = m.group(1) if m else rest
    return f"bp2_{prob}_{method}_{hp_tag}"


# ---- Problem metadata (mirrors submit scripts) ----

PROB_ARG = {
    "knapsack":           "knapsack",
    "knapsack-real":      "knapsack",
    "energy":             "energy",
    "budgetalloc":        "budgetalloc",
    "cubic":              "cubic",
    "bipartitematching":  "bipartitematching",
    "portfolio":          "portfolio",
    "asurv":              "asurv",
    "cook_county":        "cook_county",
    "speed_humps":        "speed_humps",
    "sp_synth":           "sp_synth",
    "sp_planted":         "sp_planted",
    "pg_misspec":         "pg_misspec",
    "shortestpath":       "shortestpath",
}

PROB_VERSION = {
    "knapsack":           "gen",
    "knapsack-real":      "energy",
    "energy":             "energy",
    "budgetalloc":        "real",
    "cubic":              "gen",
    "bipartitematching":  "cora",
    "portfolio":          "real",
    "asurv":              "real",
    "cook_county":        "real",
    "speed_humps":        "real",
    "sp_synth":           "synth",
    "sp_planted":         "planted",
    "pg_misspec":         "v3",
    "shortestpath":       "warcraft",
}


def out_dir(prob, opt_model, prefix):
    parg = PROB_ARG.get(prob, prob)
    pver = PROB_VERSION.get(prob, "gen")
    return os.path.join(RESULTS_ROOT, f"{parg}-{pver}", opt_model, prefix)


def cell_status(prob, opt_model, prefix, queued=None, phase=None):
    """
    Returns one of: "done", "running", "stranded", "missing".
      done      = results.npy exists
      running   = checkpoint exists AND job is in SLURM queue (needs queued+phase)
      stranded  = checkpoint exists AND job NOT in queue (timed out / failed w/ ckpt)
      missing   = no checkpoint, no results
    If queued is None, "running" and "stranded" collapse to "running" (legacy behavior).
    """
    d = out_dir(prob, opt_model, prefix)
    results = os.path.join(d, "results.npy")
    ckpt    = os.path.join(d, "checkpoints", "checkpoint_latest.pt")
    if os.path.exists(results):
        return "done"
    if not os.path.exists(ckpt):
        return "missing"
    if queued is None or phase is None:
        return "running"
    jn = prefix_to_job_name(phase, prob, opt_model, prefix)
    return "running" if jn in queued else "stranded"


def load_regret(prob, opt_model, prefix):
    """Return mean test regret or None."""
    d = out_dir(prob, opt_model, prefix)
    results = os.path.join(d, "results.npy")
    if not os.path.exists(results):
        return None
    try:
        r = np.load(results, allow_pickle=True)
        regret = np.array(r[1], dtype=float)
        return float(np.mean(regret))
    except Exception:
        return None


def load_manifest(phase):
    path = f"sweep_manifest_p{phase}.json"
    if not os.path.exists(path):
        print(f"Manifest not found: {path}")
        print("Run the submit script with --dry-run first to generate the manifest.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def print_status_grid(manifest, show_vals, phase=None, queued=None):
    """
    manifest: list of {"prob": ..., "opt_model": ..., "prefix": ...}
    Rows = methods, Cols = problems.

    If `queued` is provided, cells with a checkpoint are split into
    "running" (job in queue) vs "stranded" (ckpt exists, not in queue).
    """
    problems  = []
    methods   = []
    seen_p = set()
    seen_m = set()
    for entry in manifest:
        p = entry["prob"]
        m = entry["opt_model"]
        if p not in seen_p:
            problems.append(p)
            seen_p.add(p)
        if m not in seen_m:
            methods.append(m)
            seen_m.add(m)

    # Build lookup: (prob, method) -> list of prefix entries
    table = {}
    for entry in manifest:
        key = (entry["prob"], entry["opt_model"])
        table.setdefault(key, []).append(entry["prefix"])

    col_w = 14

    # Header
    header = f"  {'method':16s}  " + "  ".join(f"{p[:col_w]:>{col_w}}" for p in problems)
    print(header)
    print("  " + "-" * (18 + (col_w + 2) * len(problems)))

    n_done = n_run = n_strand = n_miss = 0
    show_stranded = queued is not None

    for method in methods:
        row_cells = []
        for prob in problems:
            prefixes = table.get((prob, method), [])
            if not prefixes:
                row_cells.append(f"{'N/A':>{col_w}}")
                continue

            statuses = [cell_status(prob, method, pfx, queued=queued, phase=phase) for pfx in prefixes]
            n_total = len(statuses)
            n_d = statuses.count("done")
            n_r = statuses.count("running")
            n_s = statuses.count("stranded")
            n_m = statuses.count("missing")

            n_done   += n_d
            n_run    += n_r
            n_strand += n_s
            n_miss   += n_m

            if show_vals and n_d > 0:
                regrets = [load_regret(prob, method, pfx) for pfx in prefixes]
                regrets = [r for r in regrets if r is not None]
                best = min(regrets)
                tag = f"{best:.4f}({n_d}/{n_total})"
            elif not show_stranded:
                # legacy single-R display
                n_r_combined = n_r + n_s
                if n_d == n_total:
                    tag = f"✓({n_d})"
                elif n_d > 0:
                    tag = f"✓{n_d}/R{n_r_combined}/{n_total}"
                elif n_r_combined > 0:
                    tag = f"R({n_r_combined}/{n_total})"
                else:
                    tag = f"--({n_total})"
            else:
                # 4-state display: done/running/stranded/missing
                parts = []
                if n_d: parts.append(f"✓{n_d}")
                if n_r: parts.append(f"R{n_r}")
                if n_s: parts.append(f"T{n_s}")
                if n_m: parts.append(f"-{n_m}")
                if n_d == n_total:
                    tag = f"✓({n_d})"
                else:
                    tag = "/".join(parts) + f"/{n_total}"

            row_cells.append(f"{tag:>{col_w}}")

        print(f"  {method:16s}  " + "  ".join(row_cells))

    print()
    n_total_all = n_done + n_run + n_strand + n_miss
    if show_stranded:
        print(f"  Total: {n_total_all}  ✓ done={n_done}  R running={n_run}  T stranded={n_strand}  -- missing={n_miss}")
        print("  (R = checkpoint + in SLURM queue; T = checkpoint but NOT in queue — likely TIMEOUT or error)")
    else:
        print(f"  Total: {n_total_all}  ✓ done={n_done}  R in-progress={n_run + n_strand}  -- missing={n_miss}")


def main():
    parser = argparse.ArgumentParser(description="Sweep status dashboard")
    parser.add_argument("--phase", type=int, default=None,
                        help="Phase number (1 or 2); loads sweep_manifest_p{N}.json")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest JSON (overrides --phase)")
    parser.add_argument("--vals", action="store_true",
                        help="Show best test regret per cell instead of status codes")
    parser.add_argument("--queue", action="store_true",
                        help="Query SLURM to split R into running (R) vs stranded (T) — "
                             "T = checkpoint exists but job NOT in queue, usually TIMEOUT")
    args = parser.parse_args()

    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        label = os.path.basename(args.manifest)
        phase = None
    elif args.phase:
        manifest = load_manifest(args.phase)
        label = f"Phase {args.phase}"
        phase = args.phase
    else:
        parser.print_help()
        sys.exit(1)

    queued = get_queued_job_names() if args.queue else None
    if args.queue and phase is None:
        print("Warning: --queue requires --phase to map prefixes to job names; ignoring.\n")
        queued = None

    print(f"\n=== Sweep status: {label}  ({'regret' if args.vals else 'status codes'}) ===\n")
    print_status_grid(manifest, args.vals, phase=phase, queued=queued)


if __name__ == "__main__":
    main()
