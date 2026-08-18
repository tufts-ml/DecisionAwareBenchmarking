#!/usr/bin/env python
# coding: utf-8
"""
Tests for the Berthet et al. perturbedSoftDecision implementation.

Run:  python -m pytest tests/test_perturbed_softdecision.py -v
"""

import sys
import os
import math

import torch
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root on path so openpto + reference impl are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "perturbations-differential-pytorch"))

from openpto.method.Models.perturbed import (
    perturbedSoftDecision,
    perturbedOptFunc_reinforce,
    sample_noise_with_gradients,
)
import perturbations as ref_perturbations  # reference impl


# ---------------------------------------------------------------------------
# Toy solver / objective helpers
# ---------------------------------------------------------------------------


def topk_solver_batched(costs, k=2):
    """Select top-K elements (1-hot encoding). Batched: (B, D) -> (B, D)."""
    _, idx = costs.topk(k, dim=-1)
    z = torch.zeros_like(costs)
    z.scatter_(-1, idx, 1.0)
    return z


class ToyProblem:
    """Minimal problem interface for testing."""

    def __init__(self, D=5, K=2, c_true=None):
        self.D = D
        self.K = K
        self.c_true = c_true  # (B, D) true cost coefficients

    def init_API(self):
        return {}

    def get_decision(self, Y, params=None, ptoSolver=None, **kwargs):
        """Y: (B, D) -> z: (B, D) 1-hot top-K."""
        z = topk_solver_batched(Y, self.K)
        return z, None

    def get_objective(self, Y, Z, params=None, **kwargs):
        """Linear objective: f = Y^T Z."""
        return (Y * Z).sum(dim=-1)


class ToyProblemQuadratic(ToyProblem):
    """Quadratic objective: f = Y^T Z - alpha ||Z||^2."""

    def __init__(self, D=5, K=2, alpha=0.5, c_true=None):
        super().__init__(D, K, c_true)
        self.alpha = alpha

    def get_objective(self, Y, Z, params=None, **kwargs):
        return (Y * Z).sum(dim=-1) - self.alpha * (Z ** 2).sum(dim=-1)


class ToyPtoSolver:
    """Minimal solver mock."""
    num_vars = 5
    modelSense = 1  # GRB.MAXIMIZE


# ---------------------------------------------------------------------------
# Test A: Soft decision equals manual E[z]
# ---------------------------------------------------------------------------

class TestSoftDecisionCorrectness:
    """Verify perturbedSoftDecision.forward returns the mean of discrete solutions."""

    def test_z_bar_equals_mean_z(self):
        D, K, B, N = 5, 2, 3, 50
        sigma = 0.5

        torch.manual_seed(42)
        theta = torch.randn(B, D, requires_grad=False)
        problem = ToyProblem(D, K)
        solver = ToyPtoSolver()

        # Use perturbedSoftDecision
        torch.manual_seed(0)
        z_bar = perturbedSoftDecision.apply(
            theta, solver, problem, None, N, sigma, "normal"
        )
        assert z_bar.shape == (B, D), f"Expected (B, D), got {z_bar.shape}"

        # Manually compute E[z] with same seed
        torch.manual_seed(0)
        noise, _ = sample_noise_with_gradients("normal", [N, B, D])
        perturbed_inputs = theta.unsqueeze(0) + sigma * noise
        solutions = []
        for n in range(N):
            z_n, _ = problem.get_decision(perturbed_inputs[n])
            solutions.append(z_n)
        z_manual = torch.stack(solutions, dim=0).mean(dim=0)

        torch.testing.assert_close(z_bar, z_manual, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test B: Linear objective ⇒ gradient matches REINFORCE
# ---------------------------------------------------------------------------

class TestLinearEquivalence:
    """For f = c^T z, both gradient estimators should agree."""

    def test_linear_gradient_match(self):
        D, K, B = 5, 2, 3
        N = 10000  # many samples for convergence
        sigma = 0.5

        torch.manual_seed(42)
        theta = torch.randn(B, D)
        c_true = torch.randn(B, D)

        problem = ToyProblem(D, K)
        solver = ToyPtoSolver()

        # --- Our new method: soft decision + chain rule ---
        theta_new = theta.detach().clone().requires_grad_(True)
        torch.manual_seed(0)
        z_bar = perturbedSoftDecision.apply(
            theta_new, solver, problem, None, N, sigma, "normal"
        )
        obj = (c_true * z_bar).sum(dim=-1)  # linear objective
        obj.sum().backward()
        grad_new = theta_new.grad.clone()

        # --- Reference perturbations.py ---
        theta_ref = theta.detach().clone().requires_grad_(True)
        soft_solver = ref_perturbations.perturbed(
            lambda x: topk_solver_batched(x, K),
            num_samples=N, sigma=sigma, noise="normal",
            batched=True, device=torch.device("cpu"),
        )
        torch.manual_seed(0)
        z_soft_ref = soft_solver(theta_ref)
        obj_ref = (c_true * z_soft_ref).sum(dim=-1)
        obj_ref.sum().backward()
        grad_ref = theta_ref.grad.clone()

        cos_sim = torch.nn.functional.cosine_similarity(
            grad_new.flatten().unsqueeze(0),
            grad_ref.flatten().unsqueeze(0),
        ).item()
        max_diff = (grad_new - grad_ref).abs().max().item()
        assert cos_sim > 0.99, f"Cosine sim {cos_sim:.4f} too low for linear obj"
        assert max_diff < 0.1, f"Max diff {max_diff:.4f} too high for linear obj"


# ---------------------------------------------------------------------------
# Test C: Finite-difference check for Jacobian
# ---------------------------------------------------------------------------

class TestFiniteDifference:
    """Verify Berthet Jacobian against numerical finite differences."""

    def test_fd_gradient(self):
        D, K, B = 4, 2, 1
        N = 20000  # many samples for stable E[z]
        sigma = 0.5
        h = 1e-3  # FD step

        torch.manual_seed(42)
        theta = torch.randn(B, D)
        c_true = torch.randn(B, D)
        problem = ToyProblem(D, K)
        solver = ToyPtoSolver()

        # --- Analytical gradient via perturbedSoftDecision ---
        theta_ad = theta.detach().clone().requires_grad_(True)
        torch.manual_seed(0)
        z_bar = perturbedSoftDecision.apply(
            theta_ad, solver, problem, None, N, sigma, "normal"
        )
        loss = (c_true * z_bar).sum()
        loss.backward()
        grad_ad = theta_ad.grad.clone()  # (B, D)

        # --- Numerical gradient via finite differences ---
        grad_fd = torch.zeros_like(theta)
        for d in range(D):
            theta_plus = theta.clone()
            theta_plus[0, d] += h
            torch.manual_seed(0)
            z_plus = perturbedSoftDecision.apply(
                theta_plus, solver, problem, None, N, sigma, "normal"
            )
            loss_plus = (c_true * z_plus).sum()

            theta_minus = theta.clone()
            theta_minus[0, d] -= h
            torch.manual_seed(0)
            z_minus = perturbedSoftDecision.apply(
                theta_minus, solver, problem, None, N, sigma, "normal"
            )
            loss_minus = (c_true * z_minus).sum()

            grad_fd[0, d] = (loss_plus - loss_minus) / (2 * h)

        cos_sim = torch.nn.functional.cosine_similarity(
            grad_ad.flatten().unsqueeze(0),
            grad_fd.flatten().unsqueeze(0),
        ).item()
        # FD is noisy, so be lenient
        assert cos_sim > 0.85, f"FD cosine sim {cos_sim:.4f} too low"


# ---------------------------------------------------------------------------
# Test D: End-to-end gradient flow through a model
# ---------------------------------------------------------------------------

class TestEndToEndGradientFlow:
    """Verify that loss.backward() updates model parameters."""

    def test_parameters_change(self):
        D, K, B = 5, 2, 4
        N = 10
        sigma = 0.5

        torch.manual_seed(42)
        problem = ToyProblem(D, K)
        solver = ToyPtoSolver()

        # Simple linear prediction model
        model = torch.nn.Linear(D, D, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        X = torch.randn(B, D)
        c_true = torch.randn(B, D)

        params_before = model.weight.data.clone()

        # 3 training steps
        for step in range(3):
            coeff_hat = model(X)
            z_bar = perturbedSoftDecision.apply(
                coeff_hat, solver, problem, None, N, sigma, "normal"
            )
            obj = (c_true * z_bar).sum(dim=-1)  # (B,)
            loss = obj.abs().mean()

            optimizer.zero_grad()
            loss.backward()

            # Verify gradients exist and are finite
            assert model.weight.grad is not None, "No gradient on model weights"
            assert torch.isfinite(model.weight.grad).all(), "Non-finite gradients"

            optimizer.step()

        params_after = model.weight.data.clone()
        assert not torch.allclose(params_before, params_after), \
            "Parameters didn't change after 3 training steps"


# ---------------------------------------------------------------------------
# Test E: Exact match with reference perturbations.py
# ---------------------------------------------------------------------------

class TestReferenceMatch:
    """Our perturbedSoftDecision backward should match the reference exactly."""

    def test_gradient_matches_reference(self):
        D, K, B = 5, 2, 3
        N = 500
        sigma = 0.5

        torch.manual_seed(42)
        theta = torch.randn(B, D)
        c_true = torch.randn(B, D)
        problem = ToyProblem(D, K)
        solver = ToyPtoSolver()

        # --- Our implementation ---
        theta_ours = theta.detach().clone().requires_grad_(True)
        torch.manual_seed(0)
        z_bar_ours = perturbedSoftDecision.apply(
            theta_ours, solver, problem, None, N, sigma, "normal"
        )
        loss_ours = (c_true * z_bar_ours).sum()
        loss_ours.backward()
        grad_ours = theta_ours.grad.clone()

        # --- Reference implementation ---
        theta_ref = theta.detach().clone().requires_grad_(True)
        soft_solver = ref_perturbations.perturbed(
            lambda x: topk_solver_batched(x, K),
            num_samples=N, sigma=sigma, noise="normal",
            batched=True, device=torch.device("cpu"),
        )
        torch.manual_seed(0)
        z_bar_ref = soft_solver(theta_ref)
        loss_ref = (c_true * z_bar_ref).sum()
        loss_ref.backward()
        grad_ref = theta_ref.grad.clone()

        # Forward should match exactly (same noise samples, same solver)
        torch.testing.assert_close(z_bar_ours, z_bar_ref, atol=1e-5, rtol=1e-5)

        # Backward should match exactly
        torch.testing.assert_close(grad_ours, grad_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test F: Quadratic objective produces different gradients than REINFORCE
# ---------------------------------------------------------------------------

class TestQuadraticDivergence:
    """For nonlinear objectives, the new method and REINFORCE should differ."""

    def test_quadratic_gradients_differ(self):
        D, K, B = 5, 2, 3
        N = 5000
        sigma = 0.5

        torch.manual_seed(42)
        theta = torch.randn(B, D)
        c_true = torch.randn(B, D)
        alpha = 1.0

        problem = ToyProblem(D, K)
        problem_quad = ToyProblemQuadratic(D, K, alpha)
        solver = ToyPtoSolver()

        # --- New method: soft decision + quadratic objective ---
        theta_new = theta.detach().clone().requires_grad_(True)
        torch.manual_seed(0)
        z_bar = perturbedSoftDecision.apply(
            theta_new, solver, problem, None, N, sigma, "normal"
        )
        # Quadratic objective evaluated on the soft decision
        obj_new = (c_true * z_bar).sum(dim=-1) - alpha * (z_bar ** 2).sum(dim=-1)
        obj_new.sum().backward()
        grad_new = theta_new.grad.clone()

        # --- Old REINFORCE method ---
        torch.manual_seed(0)
        noise, noise_grad = sample_noise_with_gradients("normal", [N, B, D])
        perturbed_inputs = theta.unsqueeze(0) + sigma * noise
        per_sample_objs = []
        for n in range(N):
            z_n, _ = problem.get_decision(perturbed_inputs[n])
            obj_n = problem_quad.get_objective(c_true, z_n)
            per_sample_objs.append(obj_n.detach())
        per_sample_objs = torch.stack(per_sample_objs, dim=0)  # (N, B)
        output = per_sample_objs.unsqueeze(-1)  # (N, B, 1)
        dy = torch.ones(B, 1)
        g_reinforce = torch.einsum(
            "nbd,nb->bd", noise_grad,
            torch.einsum("nbd,bd->nb", output, dy),
        )
        g_reinforce /= sigma * N

        # The two should NOT be the same
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_new.flatten().unsqueeze(0),
            g_reinforce.flatten().unsqueeze(0),
        ).item()
        # For a quadratic obj, cosine sim should be noticeably < 1
        assert cos_sim < 0.99, (
            f"Expected divergence for quadratic obj, got cosine sim {cos_sim:.4f}"
        )


# ---------------------------------------------------------------------------
# Test G: Full perturbed model forward (with ToyProblem)
# ---------------------------------------------------------------------------

class TestPerturbedModelForward:
    """Smoke test the full perturbed() model forward method."""

    def test_forward_returns_scalar_loss(self):
        from openpto.method.Models.perturbed import perturbed as PerturbedModel

        D, K, B = 5, 2, 4
        problem = ToyProblem(D, K)
        solver = ToyPtoSolver()

        model = PerturbedModel(
            ptoSolver=solver, n_samples=5, sigma=0.5, noise="normal"
        )

        theta = torch.randn(B, D, requires_grad=True)
        c_true = torch.randn(B, D)

        loss = model.forward(
            problem, theta, params=None, coeff_true=c_true, reduction="mean"
        )

        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        loss.backward()
        assert theta.grad is not None, "No gradient on coeff_hat"
        assert torch.isfinite(theta.grad).all(), "Non-finite gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
