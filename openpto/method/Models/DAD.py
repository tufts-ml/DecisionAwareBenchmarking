#!/usr/bin/env python
# coding: utf-8
"""
Decision-Aware Denoising (DAD) loss function.

Optimizes the Stein-corrected top-K / CO objective:

    J_stein = in_sample_obj  -  stein_weight * stein_bias

where stein_bias is estimated via Monte Carlo variance-gradient correction
(OS-VGC) with per-item variance-adaptive perturbations:

    δ_k = h * σ_j * ε_k,   ε_k ~ N(0, I)
    σ_j = sqrt((ŷ_j − y_j)² + ε)   ← differentiable per-item residual std

    stein_bias ≈ (1/K) Σ_k [f(z*(ŷ+δ_k)) − f(z*(ŷ))] / h

This generalises the paper's closed-form Stein formula (which applies only to
top-K problems in the K→∞ limit) to any CO problem in the benchmark via
reparameterised MC sampling — the same structural approach as the perturb
(DPO) method, but with data-driven per-item sigma instead of a fixed scalar.

Reference: Gupta, Huang, Rusmevichientong — "Decision-Aware Denoising" (2024)
"""

import torch
from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_tensor


class DecisionAwareDenoising(optModel):
    """
    Decision-Aware Denoising loss.

    Parameters
    ----------
    ptoSolver     : solver instance
    stein_weight  : float (default 1.0)
        Weight on the Stein bias correction term.  Analogous to ``lambd``
        in the blackbox method.  Swept in Phase 2.
    n_samples     : int (default 10)
        Number of MC perturbation samples for the VGC estimate.
    sigma         : float (default -1.0)
        If > 0, override per-item variance with this fixed sigma (for
        ablation studies).  -1 = use per-item residual std (recommended).
    reduction     : str (default "mean")
    """

    def __init__(
        self,
        ptoSolver,
        stein_weight=1.0,
        n_samples=10,
        sigma=-1.0,
        reduction="mean",
        **kwargs,
    ):
        super().__init__(ptoSolver)
        self.stein_weight   = float(stein_weight)
        self.n_samples      = int(n_samples)
        self.sigma_override = float(sigma)   # -1 = use residuals
        self.reduction      = reduction
        self.eps            = 1e-8

    def forward(
        self,
        problem,
        coeff_hat,
        coeff_true=None,
        params=None,
        **hyperparams,
    ):
        """
        Compute DAD loss.

        Parameters
        ----------
        coeff_hat  : Tensor (B, S, O) — neural network predictions
        coeff_true : Tensor (B, S, O) — ground-truth labels (used for σ_j)
        params     : aux data (passed through to solver)

        Returns
        -------
        Scalar loss (or per-instance losses before reduction).
        """
        device   = coeff_hat.device
        maximize = self.ptoSolver.modelSense == GRB.MAXIMIZE

        # ---- Per-item sigma (differentiable through coeff_hat) ----
        if self.sigma_override > 0 or coeff_true is None:
            # Fixed sigma override (ablation) or no ground truth available
            sigma_j = torch.full_like(coeff_hat, max(self.sigma_override, 1e-4))
        else:
            # σ_j = sqrt( (ŷ_j − y_j)² + ε )  — per-item residual std
            sigma_j = ((coeff_hat - coeff_true).pow(2) + self.eps).sqrt()

        # Bandwidth: h = S^{-1/6}  (Stein scaling from the paper)
        S = coeff_hat.shape[1]
        h = float(S) ** (-1.0 / 6.0)

        # ---- Baseline solve at ŷ (constants — no gradient through z0) ----
        c0    = coeff_hat.detach().cpu()
        z0, _ = problem.get_decision(
            c0, params, self.ptoSolver, **problem.init_API()
        )
        z0 = to_tensor(z0).to(device)

        # In-sample objective — gradient flows through coeff_hat only
        obj0 = problem.get_objective(coeff_hat, z0.detach(), params)  # (B,)

        # ---- MC Stein correction (reparameterised) ----
        correction_sum = torch.zeros_like(obj0)
        for _ in range(self.n_samples):
            eps_k   = torch.randn_like(coeff_hat)           # (B, S, O) ~ N(0,I)
            delta_k = h * sigma_j * eps_k                   # variance-adaptive perturbation

            c_pert  = (coeff_hat + delta_k).detach().cpu()  # (B, S, O), constant for solver
            zk, _   = problem.get_decision(
                c_pert, params, self.ptoSolver, **problem.init_API()
            )
            zk = to_tensor(zk).to(device)

            # Objective at perturbed point (gradient flows through coeff_hat + delta_k)
            obj_k = problem.get_objective(
                coeff_hat + delta_k, zk.detach(), params
            )  # (B,)
            correction_sum = correction_sum + (obj_k - obj0)

        # stein_bias ≈ mean_k[ (f_k − f_0) ] / h
        stein_bias = correction_sum / (self.n_samples * h)  # (B,)

        # ---- Loss = −J_stein ----
        # Maximise: J = obj0 − stein_weight * stein_bias  →  loss = −J
        # Minimise: J = −obj0 + stein_weight * stein_bias  →  loss = +J
        if maximize:
            loss = -(obj0 - self.stein_weight * stein_bias)
        else:
            loss = obj0 + self.stein_weight * stein_bias

        reduction = hyperparams.get("reduction", self.reduction)
        return do_reduction(loss, reduction)
