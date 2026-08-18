#!/usr/bin/env python
# coding: utf-8
"""
Simple Misspecification experiment from Gupta & Huang (2024), §4.1
(arXiv 2402.03256), variant 3 (asymmetric exponential noise).

Data-generating process (defaults, m=0, alpha=1):
  x ~ Uniform(0, 2)
  c(x) = -4x + 2,              for x in [0, 0.55)
         m*(x - 0.55) + c0,    for x in [0.55, 2]      # m=0 -> c=c0=-0.2 (flat)
  eps  = sqrt(alpha)*(zeta - sd) + sqrt(1 - alpha)*gamma
         zeta  ~ Exponential(scale=sd),    gamma ~ N(0, sd^2)
  Y    = c(x) + eps

Decision: z in {0, 1}; under MINIMIZE pick z=1 iff predicted cost <= 0.

Hypothesis class is linear in [1, x] (no separate bias); misspecified vs the
piecewise-linear truth. With alpha=1 noise is mean-zero asymmetric exponential
(right tail unbounded, hard floor at -sd), which breaks SPO+ Fisher consistency.
"""

import numpy as np
import torch

from gurobipy import GRB

from openpto.method.utils_method import to_array, to_device, to_tensor
from openpto.problems.PTOProblem import PTOProblem


class PGMisspec(PTOProblem):
    """PG paper Section 4.1 misspecification experiment (variant 3)."""

    def __init__(
        self,
        num_train_instances=400,
        num_test_instances=10000,
        val_frac=0.5,           # overridden internally to match paper's n_val=200
        rand_seed=2023,
        prob_version="v3",
        m=0.0,
        m0=-4.0,
        c0=-0.2,
        alpha=1.0,
        sd=0.5,
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super().__init__(data_dir)

        if prob_version != "v3":
            raise NotImplementedError(
                f"Only v3 supported; got prob_version={prob_version}"
            )

        # Force n_val = 200 per PG paper (Fig 1/Fig 3 use validation set of size 200).
        # Wrapper passes the CLI --val_frac; we override here so this task is
        # faithful regardless of the global default.
        n_val_target = 200
        if num_train_instances <= n_val_target:
            raise ValueError(
                f"num_train_instances ({num_train_instances}) must exceed n_val=200"
            )
        val_frac = float(n_val_target) / float(num_train_instances)

        self.rand_seed = rand_seed
        self.prob_version = prob_version
        self.m = float(m)
        self.m0 = float(m0)
        self.c0 = float(c0)
        self.alpha = float(alpha)
        self.sd = float(sd)
        # Kink location: solve m0*x + 2 = c0 -> x = (c0 - 2)/m0 = 0.55 with defaults.
        self.t_kink = (self.c0 - 2.0) / self.m0

        self._set_seed(rand_seed)
        N = num_train_instances + num_test_instances

        # Features: scalar x ~ Uniform(0, 2)
        x = np.random.uniform(0.0, 2.0, N).astype(np.float32)

        # Clean cost
        c_clean = np.where(
            x < self.t_kink,
            self.m0 * x + 2.0,
            self.m * (x - self.t_kink) + self.c0,
        ).astype(np.float32)

        # Mean-zero asymmetric noise
        zeta = np.random.exponential(self.sd, N).astype(np.float32) - self.sd
        gamma = np.random.normal(0.0, self.sd, N).astype(np.float32)
        eps = np.sqrt(self.alpha) * zeta + np.sqrt(1.0 - self.alpha) * gamma

        Y = (c_clean + eps).astype(np.float32)

        # Print noise moments for sanity (target mean=0, var=sd^2=0.25).
        print(
            f"[PGMisspec v3] N={N}, m={self.m}, alpha={self.alpha}, sd={self.sd}; "
            f"eps mean={eps.mean():.4f}, var={eps.var():.4f} "
            f"(target mean~0, var~{self.sd ** 2:.3f})"
        )

        # Model receives [1, x] so a linear layer (no bias) recovers (w0 + w1*x).
        feats = np.stack([np.ones(N, dtype=np.float32), x], axis=1)  # (N, 2)
        costs = Y[:, None]                                            # (N, 1)
        # Per-instance "oracle" decision under the observed (noisy) cost:
        # z* = 1 iff Y <= 0 (MINIMIZE). Stored as Y_aux for methods that want it.
        sols = (Y <= 0).astype(np.float32)[:, None]                   # (N, 1)

        # Split: [0, n_val) val, [n_val, num_train) train, [num_train:] test
        n_val = int(round(val_frac * num_train_instances))

        self.Xs_val = torch.FloatTensor(feats[:n_val])
        self.Ys_val = torch.FloatTensor(costs[:n_val])
        self.Zs_val = torch.FloatTensor(sols[:n_val])

        self.Xs_train = torch.FloatTensor(feats[n_val:num_train_instances])
        self.Ys_train = torch.FloatTensor(costs[n_val:num_train_instances])
        self.Zs_train = torch.FloatTensor(sols[n_val:num_train_instances])

        self.Xs_test = torch.FloatTensor(feats[num_train_instances:])
        self.Ys_test = torch.FloatTensor(costs[num_train_instances:])
        self.Zs_test = torch.FloatTensor(sols[num_train_instances:])

        print(
            f"[PGMisspec v3] train={len(self.Xs_train)}, val={len(self.Xs_val)}, "
            f"test={len(self.Xs_test)}, x_kink={self.t_kink:.3f}"
        )

        # Undo seed
        self._set_seed()

    # ---- PTOProblem interface ----

    def get_train_data(self, **kwargs):
        return self.Xs_train, self.Ys_train, self.Zs_train

    def get_val_data(self, **kwargs):
        return self.Xs_val, self.Ys_val, self.Zs_val

    def get_test_data(self, **kwargs):
        return self.Xs_test, self.Ys_test, self.Zs_test

    def get_model_shape(self):
        return 2, 1   # 2 input features ([1, x]), scalar output

    def get_output_activation(self):
        return "identity"

    def get_twostageloss(self):
        return "mse"

    def get_eval_metric(self):
        return "regret"

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        Z = to_device(Z, Y.device)
        return torch.sum(Y * Z, dim=-1, keepdim=True)

    def get_decision(self, Y, params, ptoSolver=None, isTrain=True, **kwargs):
        Y = to_device(Y, "cpu")
        sol_np = ptoSolver.solve(to_array(Y), **kwargs)
        sol = to_tensor(np.asarray(sol_np))
        obj = self.get_objective(Y, sol, **kwargs)
        return sol, obj

    def init_API(self):
        return {"modelSense": GRB.MINIMIZE, "n_vars": 1}
