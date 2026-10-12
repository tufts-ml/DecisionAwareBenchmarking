#!/usr/bin/env python
# coding: utf-8
"""
Regression tests for the declared optimisation sense of the Cubic Top-K task.

Every task declares, via ``init_API()['modelSense']``, whether its objective is
maximised or minimised; the decision-aware losses (SPO+, NCE, PG, DFL, DAD, the
pairwise/listwise LTR losses, ...) branch on that declaration. The TopKSolver
that Cubic Top-K is paired with picks the K LARGEST predicted values and never
consults the declaration. Upstream (Thinklab-SJTU/PredictiveCO-Benchmark,
f885426) declared this task as a MINIMIZE problem, so those losses trained
with the wrong sign on it; the fix is a one-line change in
``openpto/problems/CubicTopK.py`` (see REPRODUCING.md, "Cubic Top-K sense
correction"). These tests pin the corrected behaviour and keep the legacy
configuration reproducible.

Run:  python -m pytest tests/test_solver_sense.py -v
"""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from gurobipy import GRB  # noqa: E402  pylint: disable=no-name-in-module

from openpto.method.Models.SPO import SPO  # noqa: E402
from openpto.method.Solvers.heuristic.TopKSolver import TopKSolver  # noqa: E402
from openpto.problems.CubicTopK import CubicTopK  # noqa: E402

LEGACY_CONFIG = os.path.join(
    PROJECT_ROOT, "openpto", "config", "probs", "cubic_legacy_minsense.yaml"
)


@pytest.fixture(scope="module")
def cubic():
    """A small Cubic Top-K instance (same DGP as the benchmark, fewer items)."""
    torch.manual_seed(0)
    prob = CubicTopK(
        num_train_instances=40,
        num_test_instances=40,
        num_items=20,
        budget=3,
        rand_seed=2023,
    )
    prob.device = torch.device("cpu")
    return prob


def _solver_direction(prob, solver, Y, n_alternatives=200):
    """Infer which direction ``solver`` optimises ``prob.get_objective`` in.

    Scores the solver's own decision and many random feasible decisions
    (uniform K-subsets) under the same true costs. If no alternative beats the
    solver's decision the solver MAXIMISES; if none is worse it MINIMISES.
    """
    B, n_items = Y.shape[0], Y.shape[1]
    z_star = solver.solve(Y, prob.budget)
    base = prob.get_objective(Y, z_star)
    g = torch.Generator().manual_seed(0)
    better = worse = 0
    for _ in range(n_alternatives):
        idx = torch.stack([torch.randperm(n_items, generator=g)[: prob.budget] for _ in range(B)])
        z_alt = torch.nn.functional.one_hot(idx, n_items).sum(dim=1).to(Y.dtype)
        alt = prob.get_objective(Y, z_alt)
        better += int((alt > base + 1e-6).sum())
        worse += int((alt < base - 1e-6).sum())
    assert better + worse > 0, "degenerate instance: every K-subset ties"
    if better == 0:
        return GRB.MAXIMIZE
    if worse == 0:
        return GRB.MINIMIZE
    raise AssertionError(
        f"solver is not optimal in either direction ({better} better, {worse} worse)"
    )


def test_cubic_declares_maximize(cubic):
    api = cubic.init_API()
    assert api["modelSense"] == GRB.MAXIMIZE
    assert api["n_vars"] == cubic.Ys_train.shape[1]


def test_declared_sense_matches_topk_solver(cubic):
    """The sense CubicTopK declares must be the one TopKSolver actually optimises."""
    api = cubic.init_API()
    solver = TopKSolver(**api)
    _, Y, _ = cubic.get_test_data()
    actual = _solver_direction(cubic, solver, Y[:16])
    assert actual == api["modelSense"], (
        "CubicTopK.init_API declares a different sense than its solver optimises; "
        "sense-branching losses would train with the wrong sign."
    )


def test_spo_plus_loss_nonnegative_under_declared_sense(cubic):
    """SPO+ is a non-negative upper bound on regret -- but only when the sense is right.

    Under the upstream MINIMIZE declaration the very same loss is <= 0 on this
    task (its sign is flipped), which is how the bug surfaced in training logs.
    """
    _, Y, _ = cubic.get_train_data()
    Y = Y[:16]
    torch.manual_seed(1)
    Y_hat = Y + 0.5 * torch.randn_like(Y)

    solver = TopKSolver(**cubic.init_API())
    loss = SPO(solver).forward(cubic, Y_hat, Y, None, reduction="mean")
    assert float(loss) >= -1e-6, f"SPO+ loss negative under declared sense: {float(loss)}"
    assert float(loss) > 1e-3, "noisy predictions should incur a strictly positive SPO+ loss"

    legacy = TopKSolver(modelSense=GRB.MINIMIZE, n_vars=cubic.init_API()["n_vars"])
    legacy_loss = SPO(legacy).forward(cubic, Y_hat, Y, None, reduction="mean")
    assert float(legacy_loss) < 0, (
        "legacy (upstream) MINIMIZE declaration should flip the SPO+ sign; "
        f"got {float(legacy_loss)}"
    )


def test_legacy_config_restores_upstream_sense(cubic):
    """cubic_legacy_minsense.yaml overrides modelSense the way solver_wrapper merges it."""
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe", pure=True)
    with open(LEGACY_CONFIG) as f:
        conf = yaml.load(f)
    assert conf["solver"]["heuristic"]["modelSense"] == GRB.MINIMIZE == 1
    # Mirrors openpto/method/Solvers/wrapper_solver.py:
    #   solve_dict = {**problem.init_API(), **conf["solver"][args.solver]}
    solve_dict = {**cubic.init_API(), **conf["solver"]["heuristic"]}
    solver = TopKSolver(**solve_dict)
    assert solver.modelSense == GRB.MINIMIZE
    # ... and the default config leaves the corrected declaration in place.
    with open(os.path.join(PROJECT_ROOT, "openpto", "config", "probs", "cubic.yaml")) as f:
        default_conf = yaml.load(f)
    default_solver = TopKSolver(**{**cubic.init_API(), **default_conf["solver"]["heuristic"]})
    assert default_solver.modelSense == GRB.MAXIMIZE
