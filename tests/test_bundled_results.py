#!/usr/bin/env python
# coding: utf-8
"""
Consistency checks on the bundled result JSONs in results/.

The bundled files are the values the manuscript reports. In particular the
Cubic Top-K cells of the eight sense-dependent methods come from the
corrected re-run (REPRODUCING.md, "Cubic Top-K sense correction"), while
results/bench_p2_best_val_oldsense.json keeps the pre-correction values for
the sense-fix comparison table. These tests guard against re-exporting a
stale or over-patched variant.

Run:  python -m pytest tests/test_bundled_results.py -v
"""

import json
import os

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")

CUBICFIX_METHODS = ["spo", "nce", "identity", "blackbox", "pg", "dfl", "pairLTR", "listLTR"]
ENERGYVR_METHODS = ["mse", "dfl", "identity"]
PAPER_TASKS = ["knapsack", "knapsack-real", "energy", "budgetalloc", "cubic", "bipartitematching",
               "asurv", "cook_county", "speed_humps", "sp_synth", "pg_misspec", "shortestpath"]


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _cells(d):
    return {(m, t) for m in d for t in d[m]}


def test_p2_differs_from_oldsense_only_in_corrected_cubic_and_energy_cells():
    new, old = load("bench_p2_best_val.json"), load("bench_p2_best_val_oldsense.json")
    assert _cells(new) == _cells(old)
    differing = {(m, t) for (m, t) in _cells(new)
                 if json.dumps(new[m][t], sort_keys=True) != json.dumps(old[m][t], sort_keys=True)}
    assert differing == ({(m, "cubic") for m in CUBICFIX_METHODS}
                         | {(m, "energy") for m in ENERGYVR_METHODS})


def test_energy_pto_cells_come_from_the_solver_eval_rerun():
    p2, p1, lm = load("bench_p2_best_val.json"), load("bench_p1_best_val.json"), load("loss_matrix.json")
    for m in ENERGYVR_METHODS:
        assert p2[m]["energy"]["family"] == "bench_energyvr"
        assert p2[m]["energy"]["prefix"].startswith("bench_energyvr_")
        assert p2[m]["energy"]["phase"] == 1 and p2[m]["energy"]["hp_name"] is None
        assert (p1[m]["energy"]["lr"], p1[m]["energy"]["batch"]) == (p2[m]["energy"]["lr"], p2[m]["energy"]["batch"])
        assert lm["energy"][m]["prefix"] == p2[m]["energy"]["prefix"]
        # in-training decision regret exists (the legacy runs skipped the solver)
        assert lm["energy"][m]["metrics"]["val_regret"] is not None


def test_corrected_cubic_cells_come_from_the_corrected_run_family():
    p2, p1, lm = load("bench_p2_best_val.json"), load("bench_p1_best_val.json"), load("loss_matrix.json")
    for m in CUBICFIX_METHODS:
        assert p2[m]["cubic"]["family"] == "bench_cubicfix"
        assert p2[m]["cubic"]["prefix"].startswith("bench_cubicfix_p")
        assert lm["cubic"][m]["prefix"] == p2[m]["cubic"]["prefix"]
        # Phase-1 pick (lr, batch) is the one Phase 2 was swept at
        assert (p1[m]["cubic"]["lr"], p1[m]["cubic"]["batch"]) == (p2[m]["cubic"]["lr"], p2[m]["cubic"]["batch"])
    for m in ("mse", "pointLTR", "lodl", "perturb"):  # sense-independent: untouched
        assert "family" not in p2[m]["cubic"]


def test_spo_plus_on_cubic_is_corrected():
    new, old = load("bench_p2_best_val.json"), load("bench_p2_best_val_oldsense.json")
    assert new["spo"]["cubic"]["test"] < 0.001       # 0.01% relative regret
    assert old["spo"]["cubic"]["test"] > 1.0         # 124% under the inverted sense


def test_rank_counterfactual_cache_covers_paper_tasks_in_mean_mode():
    cache = load("rank_counterfactual_cache.json")
    missing = [t for t in PAPER_TASKS if f"t|mean|{t}" not in cache]
    assert not missing, missing
    # only the mean-mode scans of cubic / energy were redone after the correction / re-run
    for t in ("cubic", "energy"):
        assert f"t|best|{t}" not in cache and f"t|seed0|{t}" not in cache


def test_bump_rerun_data_carries_corrected_cubic_cells():
    # The eight corrected cubic cells were spliced in as the 10-seed means of
    # bench_p2_best_val.json (in percent); the other cells are the single-seed
    # re-run values the figure was originally built from (see REPRODUCING.md).
    bb, p2 = load("bench_bump_rerun_data.json"), load("bench_p2_best_val.json")
    disp = bb["method_display"]
    for m in CUBICFIX_METHODS:
        assert bb["regret"]["cubic"][disp[m]] == pytest.approx(100 * p2[m]["cubic"]["test"], abs=1e-4)


def test_appendix_inputs_present():
    for name in ("geng_reproduction_data.json", "runtimes_per_method_task.json",
                 "bench_fullgrid_best_val.json"):
        assert os.path.exists(os.path.join(RESULTS, name)), name
