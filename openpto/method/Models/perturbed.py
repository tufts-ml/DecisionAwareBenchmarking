#!/usr/bin/env python
# coding: utf-8
"""
Differentiable Perturbed Optimizers  (Berthet et al., NeurIPS 2020)

The default ``perturbed`` class implements the *correct* DPO approach:
  forward  — returns the **soft decision** E_ε[z(θ̂ + σε)]
  backward — Jacobian–vector product via the log-derivative trick on the
             **solution vectors** (not scalar objectives).
  loss     — evaluated externally: |f(θ*, z̄) − f(θ*, z*)|

Reference PyTorch implementation:
  https://github.com/tuero/perturbations-differential-pytorch
"""

import logging
import math

import numpy as np
import torch
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise helpers (ported from reference perturbations.py)
# ---------------------------------------------------------------------------

_GUMBEL = "gumbel"
_NORMAL = "normal"
SUPPORTED_NOISES = (_GUMBEL, _NORMAL)


def sample_noise_with_gradients(noise, shape):
    """Sample noise and the gradient of log p(noise).

    For Normal noise the gradient equals the noise itself.
    For Gumbel noise the gradient is ``1 - exp(-noise)``.
    """
    if noise not in SUPPORTED_NOISES:
        raise ValueError(
            f"{noise} noise is not supported. Use one of {SUPPORTED_NOISES}"
        )
    if noise == _GUMBEL:
        sampler = Gumbel(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = 1 - torch.exp(-samples)
    elif noise == _NORMAL:
        sampler = Normal(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = samples
    return samples, gradients


# ---------------------------------------------------------------------------
# Sigma Scheduler
# ---------------------------------------------------------------------------

class SigmaScheduler:
    """
    Schedule the perturbation amplitude (sigma) over training epochs.

    Supported schedules:
        constant     – sigma is fixed (default behaviour).
        linear_decay – linear interpolation from sigma_start to sigma_end.
        cosine_decay – cosine annealing from sigma_start to sigma_end.
        step_decay   – multiply by gamma every step_size epochs.

    Usage:
        sched = SigmaScheduler("cosine_decay", sigma_start=1.0, sigma_end=0.01,
                               n_epochs=300)
        for epoch in range(1, 301):
            sigma = sched.step(epoch)
    """

    def __init__(
        self,
        schedule="constant",
        sigma_start=1.0,
        sigma_end=0.01,
        n_epochs=300,
        sigma_gamma=0.5,
        sigma_step_size=100,
        warmup_epochs=0,
    ):
        self.schedule = schedule
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.n_epochs = max(n_epochs, 1)
        self.gamma = float(sigma_gamma)
        self.step_size = int(sigma_step_size)
        self.warmup_epochs = int(warmup_epochs)
        self.current_sigma = self.sigma_start

    def step(self, epoch):
        """Return the sigma value for the given *1-based* epoch.

        If ``warmup_epochs > 0``, sigma is held at ``sigma_start`` for the
        first ``warmup_epochs`` epochs; decay begins at epoch
        ``warmup_epochs + 1`` and runs over the remaining
        ``n_epochs - warmup_epochs`` epochs.
        """
        # During warmup, hold sigma constant at sigma_start
        if epoch <= self.warmup_epochs:
            self.current_sigma = self.sigma_start
            return self.current_sigma

        # Effective epoch and total for the decay phase
        decay_epoch = epoch - self.warmup_epochs
        decay_total = max(self.n_epochs - self.warmup_epochs, 1)

        if self.schedule == "constant":
            self.current_sigma = self.sigma_start
        elif self.schedule == "linear_decay":
            frac = min(decay_epoch / decay_total, 1.0)
            self.current_sigma = self.sigma_start + frac * (self.sigma_end - self.sigma_start)
        elif self.schedule == "cosine_decay":
            frac = min(decay_epoch / decay_total, 1.0)
            self.current_sigma = (
                self.sigma_end
                + 0.5 * (self.sigma_start - self.sigma_end) * (1 + math.cos(math.pi * frac))
            )
        elif self.schedule == "step_decay":
            n_steps = decay_epoch // self.step_size
            self.current_sigma = self.sigma_start * (self.gamma ** n_steps)
        else:
            raise ValueError(f"Unknown sigma schedule: {self.schedule}")
        return self.current_sigma

    def __repr__(self):
        warmup_str = f", warmup={self.warmup_epochs}" if self.warmup_epochs > 0 else ""
        return (
            f"SigmaScheduler(schedule={self.schedule}, start={self.sigma_start}, "
            f"end={self.sigma_end}, n_epochs={self.n_epochs}{warmup_str})"
        )


# ---------------------------------------------------------------------------
# Perturbed Optimization Model  (Berthet et al. — correct DPO)
# ---------------------------------------------------------------------------

class perturbed(optModel):
    """
    Differentiable Perturbed Optimizer (Berthet et al., NeurIPS 2020).

    Produces a **soft decision** z̄ = E_ε[z(θ̂ + σε)] via ``perturbedSoftDecision``
    and evaluates the objective under true coefficients.  Gradients flow through
    z̄ back to the prediction model via the Berthet Jacobian.

    Loss = |f(θ*, z̄) − f(θ*, z*)|   (direction-agnostic regret)
    """

    def __init__(
        self,
        ptoSolver,
        n_samples=10,
        sigma=1.0,
        noise="normal",
        seed=135,
        output_activation="none",
        loss_type="regret",
        # Sigma scheduler params (optional)
        sigma_schedule="constant",
        sigma_start=None,
        sigma_end=0.01,
        sigma_n_epochs=300,
        sigma_gamma=0.5,
        sigma_step_size=100,
        sigma_warmup_epochs=0,
        **hyperparams,
    ):
        """
        Args:
            ptoSolver (optModel): an  optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            noise (str): noise distribution, 'normal' or 'gumbel'
            seed (int): random state seed
            output_activation (str): activation applied to predictions before
                perturbation. One of 'none', 'sigmoid', 'tanh', 'softplus'.
            sigma_schedule (str): 'constant', 'linear_decay', 'cosine_decay', 'step_decay'
            sigma_start (float): starting sigma (defaults to sigma if None)
            sigma_end (float): ending sigma for decay schedules
            sigma_n_epochs (int): total epochs for schedule computation
            sigma_gamma (float): multiplicative factor for step_decay
            sigma_step_size (int): epoch interval for step_decay
            sigma_warmup_epochs (int): hold sigma_start for this many epochs
                before starting the decay schedule (default: 0)
        """
        super().__init__(ptoSolver)
        # Loss type: "regret" (default — extra solver call) or "objective" (direct objective)
        if loss_type not in ("regret", "objective"):
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. Choose 'regret' or 'objective'."
            )
        self.loss_type = loss_type
        self.model_sense = ptoSolver.modelSense  # GRB.MAXIMIZE or GRB.MINIMIZE
        self.n_samples = n_samples
        self.sigma = sigma
        if noise not in SUPPORTED_NOISES:
            raise ValueError(
                f"{noise} noise not supported. Use one of {SUPPORTED_NOISES}"
            )
        self.noise = noise
        self.rnd = np.random.RandomState(seed)
        n_vars = ptoSolver.num_vars
        self.solpool = np.empty((0, n_vars), dtype=np.float64)

        # Output activation
        _act_map = {
            "none": lambda x: x,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "softplus": torch.nn.functional.softplus,
        }
        if output_activation not in _act_map:
            raise ValueError(
                f"Unknown output_activation '{output_activation}'. "
                f"Choose from {list(_act_map.keys())}"
            )
        self.output_activation = _act_map[output_activation]

        # Per-epoch scale tracking
        self._batch_scales = []

        # Per-epoch loss component tracking (for gradient scale diagnostics)
        self._batch_perturb_losses = []   # raw perturb loss values per batch
        self._batch_pred_mse_losses = []  # raw MSE-reg loss values per batch (before λ scaling)

        # Sigma scheduler
        if sigma_start is None:
            sigma_start = sigma
        self.sigma_scheduler = SigmaScheduler(
            schedule=sigma_schedule,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            n_epochs=sigma_n_epochs,
            sigma_gamma=sigma_gamma,
            sigma_step_size=sigma_step_size,
            warmup_epochs=sigma_warmup_epochs,
        )

    def step(self, epoch):
        """Update sigma according to the schedule. Called by ExpManager each epoch."""
        if self._batch_scales:
            avg_scale = sum(self._batch_scales) / len(self._batch_scales)
            ratio = self.sigma / avg_scale if avg_scale > 0 else float('inf')
            log_msg = (
                f"  [perturb] epoch {epoch}: "
                f"|coeff_hat| = {avg_scale:.4f}, "
                f"sigma = {self.sigma:.4f}, "
                f"sigma/|coeff_hat| = {ratio:.4f}"
            )
            if self._batch_perturb_losses:
                avg_perturb = sum(self._batch_perturb_losses) / len(self._batch_perturb_losses)
                log_msg += f", L_perturb = {avg_perturb:.5f}"
            if self._batch_pred_mse_losses:
                avg_mse = sum(self._batch_pred_mse_losses) / len(self._batch_pred_mse_losses)
                # Retrieve pred_loss_weight from last forward call if available
                lam = getattr(self, "_last_pred_loss_weight", 0.0)
                log_msg += f", L_mse = {avg_mse:.5f}, lambda*L_mse = {lam * avg_mse:.5f}"
                if avg_perturb > 0:
                    log_msg += f", ratio(lambda*mse/perturb) = {lam * avg_mse / avg_perturb:.3f}"
            logger.info(log_msg)
            self._batch_scales = []
            self._batch_perturb_losses = []
            self._batch_pred_mse_losses = []
        self.sigma = self.sigma_scheduler.step(epoch)
        return self.sigma

    def forward(
        self,
        problem,
        coeff_hat,
        params,
        coeff_true=None,
        **hyperparams,
    ):
        """
        Forward pass — Berthet soft-decision formulation.

        1. Compute z̄ = E_ε[z(θ̂ + σε)]  via perturbedSoftDecision  (differentiable)
        2. Evaluate obj  = f(θ*, z̄)
        3. Evaluate obj* = f(θ*, z*)    (detached constant)
        4. loss = |obj − obj*|           (direction-agnostic regret)

        Gradients flow: loss → obj → z̄ → perturbedSoftDecision.backward → coeff_hat
        """
        # Handle list inputs (problems with variable-size instances)
        if isinstance(coeff_hat, list):
            losses = []
            for i in range(len(coeff_hat)):
                ch_i = coeff_hat[i].unsqueeze(0)
                ct_i = coeff_true[i].unsqueeze(0)
                loss_i = self.forward(
                    problem, ch_i, params, ct_i, **hyperparams
                )
                losses.append(loss_i)
            return do_reduction(torch.stack(losses), hyperparams["reduction"])

        # Apply output activation
        coeff_hat = self.output_activation(coeff_hat)

        # Track characteristic scale for diagnostics
        with torch.no_grad():
            self._batch_scales.append(coeff_hat.abs().mean().item())

        if self.sigma == 0:
            # --- σ=0 shortcut: direct differentiable solve (no perturbation) ---
            # For solvers backed by CvxpyLayer (e.g. Portfolio), gradients flow
            # through the QP's KKT conditions via implicit differentiation.
            # This is the exact σ→0 limit of the Berthet soft decision.
            # NOTE: only works for differentiable solvers; Gurobi-backed solvers
            # will produce zero gradients.
            z_bar, _ = problem.get_decision(
                coeff_hat, params, self.ptoSolver, **problem.init_API()
            )
            if not isinstance(z_bar, torch.Tensor):
                z_bar = torch.as_tensor(
                    z_bar, device=coeff_hat.device, dtype=coeff_hat.dtype
                )
            else:
                z_bar = z_bar.to(device=coeff_hat.device, dtype=coeff_hat.dtype)
        else:
            # --- Soft decision z̄ = E[z(θ̂ + σε)] --- (differentiable via Berthet Jacobian)
            z_bar = perturbedSoftDecision.apply(
                coeff_hat,
                self.ptoSolver,
                problem,
                params,
                self.n_samples,
                self.sigma,
                self.noise,
            )

        # --- Objective under true coefficients ---
        obj = problem.get_objective(coeff_true, z_bar, params)
        if not isinstance(obj, torch.Tensor):
            obj = torch.as_tensor(obj, dtype=coeff_hat.dtype)
        obj = obj.to(device=coeff_hat.device, dtype=coeff_hat.dtype)

        if self.loss_type == "objective":
            # --- Direct objective loss (no extra solver call) ---
            # For maximization: loss = -obj (we want to maximize obj)
            # For minimization: loss =  obj (we want to minimize obj)
            if self.model_sense == GRB.MAXIMIZE:
                loss = do_reduction(-obj, hyperparams["reduction"])
            else:
                loss = do_reduction(obj, hyperparams["reduction"])
        else:
            # --- Optimal objective f(θ*, z*) — constant, no gradient ---
            coeff_true_cpu = coeff_true.detach().cpu()
            z_star_np, _ = problem.get_decision(
                coeff_true_cpu, params, self.ptoSolver, **problem.init_API()
            )
            z_star = torch.as_tensor(
                z_star_np, device=coeff_hat.device, dtype=coeff_hat.dtype
            )
            obj_star = problem.get_objective(coeff_true, z_star, params)
            if not isinstance(obj_star, torch.Tensor):
                obj_star = torch.as_tensor(obj_star, dtype=coeff_hat.dtype)
            obj_star = obj_star.to(device=coeff_hat.device, dtype=coeff_hat.dtype).detach()

            # --- Direction-agnostic regret ---
            regret = torch.abs(obj - obj_star)

            loss = do_reduction(regret, hyperparams["reduction"])

        # --- Optional prediction-loss regularization ---
        # λ · MSE(θ̂, θ*) penalises the predictor for drifting away from the
        # true coefficients, which keeps |θ̂| in check and mitigates the
        # soft-to-hard gap observed on portfolio.
        pred_loss_weight = float(hyperparams.get("pred_loss_weight", 0.0))
        self._last_pred_loss_weight = pred_loss_weight
        # Track perturb loss value for gradient scale diagnostics
        self._batch_perturb_losses.append(loss.item())
        if pred_loss_weight > 0.0 and coeff_true is not None:
            pred_mse = torch.nn.functional.mse_loss(coeff_hat, coeff_true)
            self._batch_pred_mse_losses.append(pred_mse.item())
            loss = loss + pred_loss_weight * pred_mse

        return loss


class perturbedSoftDecision(torch.autograd.Function):
    """
    Autograd function for Differentiable Perturbed Optimizers (Berthet et al.).

    Forward:  z̄ = (1/N) Σ_n z(θ̂ + σε_n)     — expected solution vector
    Backward: Jacobian–vector product via the log-derivative trick:
              g_d = (1/Nσ) Σ_n (z_n · dy) · ∇log p(ε_n)_d

    This matches the reference implementation exactly.
    """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        ptoSolver,
        problem,
        params,
        n_samples,
        sigma,
        noise_type,
    ):
        """
        Args:
            coeff_hat:   (B, D1, ..., Dk)  predicted cost coefficients
            ptoSolver:   combinatorial solver wrapper
            problem:     problem instance (get_decision, init_API)
            params:      auxiliary params for solver
            n_samples:   number of Monte-Carlo perturbation samples
            sigma:       perturbation amplitude
            noise_type:  'normal' or 'gumbel'
        Returns:
            z_bar: (B, sol_D1, ...)  expected solution E[z(θ̂ + σε)]
        """
        device = coeff_hat.device
        dtype = coeff_hat.dtype
        original_input_shape = coeff_hat.shape          # (B, D1, ..., Dk)

        # ----- Sample noise -----
        perturbed_input_shape = [n_samples] + list(original_input_shape)
        additive_noise, noise_gradient = sample_noise_with_gradients(
            noise_type, perturbed_input_shape
        )
        additive_noise = additive_noise.to(device=device, dtype=dtype)
        noise_gradient = noise_gradient.to(device=device, dtype=dtype)

        # ----- Perturbed inputs: (N, B, D1, ..., Dk) -----
        perturbed_input = coeff_hat.unsqueeze(0) + sigma * additive_noise

        # ----- Solve each sample -----
        # Loop over N (not flat N*B) because some problems have aux params
        # batched at size B that cannot be expanded to N*B.
        init_api = problem.init_API()
        perturbed_solutions = []
        for n in range(n_samples):
            perturbed_n_cpu = perturbed_input[n].detach().cpu()
            sols_n, _ = problem.get_decision(
                perturbed_n_cpu, params, ptoSolver, **init_api
            )
            if not isinstance(sols_n, torch.Tensor):
                sols_n = torch.as_tensor(sols_n, device=device, dtype=dtype)
            else:
                sols_n = sols_n.to(device=device, dtype=dtype)
            perturbed_solutions.append(sols_n)
        perturbed_solutions = torch.stack(perturbed_solutions, dim=0)  # (N, B, sol_D...)

        # ----- Expected solution (soft decision) -----
        z_bar = perturbed_solutions.mean(dim=0)                        # (B, sol_D...)

        # ----- Save for backward -----
        ctx.save_for_backward(perturbed_solutions, noise_gradient)
        ctx.n_samples = n_samples
        ctx.sigma = sigma
        ctx.original_input_shape = original_input_shape
        return z_bar

    @staticmethod
    def backward(ctx, dy):
        """
        Berthet et al. Jacobian–vector product via log-derivative trick.

        perturbed_solutions:  (N, B, sol_D...)   solution vectors
        noise_gradient:       (N, B, D1, ..., Dk)  ∇log p(ε)
        dy:                   (B, sol_D...)       upstream gradient from loss

        g_d = (1/Nσ) Σ_n (z_n · dy) · ∇log p(ε_n)_d

        Note: solution dim and input dim may differ (e.g. BipartiteMatching).
        The einsum bridges the two spaces via the scalar score (z_n · dy).
        """
        perturbed_solutions, noise_gradient = ctx.saved_tensors
        n_samples = ctx.n_samples
        sigma = ctx.sigma
        original_input_shape = ctx.original_input_shape

        # Flatten spatial dims for einsum
        # solutions: (N, B, D_sol_flat),  noise_grad: (N, B, D_in_flat)
        flatten = lambda t: t.reshape(t.shape[0], t.shape[1], -1)
        sol_flat = flatten(perturbed_solutions)     # (N, B, D_sol)
        noise_grad_flat = flatten(noise_gradient)   # (N, B, D_in)
        dy_flat = dy.reshape(dy.shape[0], -1)       # (B, D_sol)

        # Score each sample: how much does z_n align with the upstream gradient?
        scores = torch.einsum("nbd,bd->nb", sol_flat, dy_flat)  # (N, B)

        # Weight noise gradients by scores → gradient w.r.t. input
        g = torch.einsum("nbd,nb->bd", noise_grad_flat, scores)  # (B, D_in)
        g /= sigma * n_samples

        g = g.reshape(original_input_shape)
        # 7 inputs to forward: coeff_hat + 6 non-tensor args
        return g, None, None, None, None, None, None
