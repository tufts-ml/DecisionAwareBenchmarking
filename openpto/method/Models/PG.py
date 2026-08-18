#!/usr/bin/env python
# coding: utf-8
"""
Perturbation Gradient (PG) loss function.

Finite-difference surrogate loss using the ground-truth cost vector as the
perturbation direction (one-sided, backward differencing).

Reference: https://arxiv.org/abs/2402.03256
"""

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_tensor


class perturbationGradient(optModel):
    """
    Perturbation Gradient loss.

    For each instance, solves the CO problem at both c_hat and (c_hat - sigma*c_true),
    then forms a finite-difference approximation of the objective gradient:

        loss = (obj(c_hat - sigma*c_true) - obj(c_hat)) / sigma   [MAXIMIZE]
        loss = (obj(c_hat) - obj(c_hat - sigma*c_true)) / sigma   [MINIMIZE]

    Gradient flows through the differentiable obj = c_hat · sol terms via PyTorch
    autograd; the solutions are treated as constants (detached from the graph).
    """

    def __init__(self, ptoSolver, **kwargs):
        super().__init__(ptoSolver)
        self.sigma = float(kwargs.get("sigma", 0.1))

    def forward(self, problem, coeff_hat, coeff_true, params=None, **hyperparams):
        # Ensure batch dimension
        if coeff_hat.dim() == 1:
            coeff_hat = coeff_hat.unsqueeze(0)
            coeff_true = coeff_true.unsqueeze(0)

        device = coeff_hat.device
        cp = coeff_hat.detach().cpu()
        c  = coeff_true.detach().cpu()

        # Solve at c_hat and at (c_hat - sigma * c_true)
        sols,  _ = problem.get_decision(
            cp, params, self.ptoSolver, **problem.init_API()
        )
        solsm, _ = problem.get_decision(
            cp - self.sigma * c, params, self.ptoSolver, **problem.init_API()
        )

        sol  = to_tensor(sols ).to(device)
        solm = to_tensor(solsm).to(device)

        # Differentiable objectives via problem.get_objective.
        # Solutions are constants; gradients flow through coeff_hat only.
        # Handles both linear objectives (knapsack etc.) and nonlinear ones
        # (budgetalloc: P(>=1 click) = 1 - prod(1 - p_i * x_i)).
        obj  = problem.get_objective(coeff_hat,                            sol.detach(),  params).to(device)
        objm = problem.get_objective(coeff_hat - self.sigma * coeff_true,  solm.detach(), params).to(device)

        if self.ptoSolver.modelSense == GRB.MAXIMIZE:
            loss = (objm - obj) / self.sigma
        elif self.ptoSolver.modelSense == GRB.MINIMIZE:
            loss = (obj - objm) / self.sigma
        else:
            raise NotImplementedError(f"Unknown modelSense: {self.ptoSolver.modelSense}")

        return do_reduction(loss, hyperparams["reduction"])
