#!/usr/bin/env python
# coding: utf-8
"""
Consistency checks on the bundled result JSONs in results/.

Run:  python -m pytest tests/test_bundled_results.py -v
"""

import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "results")

PAPER_TASKS = ["knapsack", "knapsack-real", "energy", "budgetalloc", "cubic", "bipartitematching",
               "asurv", "cook_county", "speed_humps", "sp_synth", "pg_misspec", "shortestpath"]


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def test_phase2_picks_sit_at_the_phase1_lr_and_batch():
    p1, p2 = load("bench_p1_best_val.json"), load("bench_p2_best_val.json")
    for m in p2:
        for t in p2[m]:
            assert (p2[m][t]["lr"], p2[m][t]["batch"]) == (p1[m][t]["lr"], p1[m][t]["batch"]), (m, t)


def test_loss_matrix_rows_are_the_phase2_picks():
    p2, lm = load("bench_p2_best_val.json"), load("loss_matrix.json")
    for t in lm:
        for m in lm[t]:
            if m in p2 and t in p2[m]:
                assert lm[t][m]["prefix"] == p2[m][t]["prefix"], (m, t)


def test_rank_counterfactual_cache_covers_paper_tasks():
    cache = load("rank_counterfactual_cache.json")
    missing = [t for t in PAPER_TASKS if f"t|mean|{t}" not in cache]
    assert not missing, missing


def test_appendix_table_inputs_present():
    for name in ("geng_reproduction_data.json", "runtimes_per_method_task.json",
                 "bench_fullgrid_best_val.json", "cubic_upstream_sense.json"):
        assert os.path.exists(os.path.join(RESULTS, name)), name
