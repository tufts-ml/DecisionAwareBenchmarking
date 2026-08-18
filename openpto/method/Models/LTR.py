#!/usr/bin/env python
# coding: utf-8
"""
Learning to rank Losses
"""

import numpy as np
import torch
import torch.nn.functional as F

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction, to_tensor


class pointwiseLTR(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        # solution pool
        n_vars = ptoSolver.num_vars
        self.solpool = np.empty((0, n_vars), dtype=np.float32)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """

        # coeff_hat = coeff_hat.squeeze(-1)
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux,
                ptoSolver=self.ptoSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # solve
        sol_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, self.ptoSolver, **problem.init_API()
        )
        # add into solpool
        self.solpool = np.concatenate((self.solpool, sol_hat))
        # remove duplicate
        self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = to_tensor(self.solpool).to(coeff_hat.device)
        K = solpool.shape[0]
        # Compute per-instance pointwise loss: each instance's predicted coeff is
        # scored against all K pooled solutions and compared to true coeff scores.
        losses_per_instance = []
        for i in range(coeff_hat.shape[0]):
            ch_i = coeff_hat[i:i+1].expand(K, *coeff_hat.shape[1:])
            ct_i = coeff_true[i:i+1].expand(K, *coeff_true.shape[1:])
            params_i = params[i:i+1].expand(K, *params.shape[1:]) if isinstance(params, torch.Tensor) else params
            obj_c = problem.get_objective(ct_i, solpool, params_i)
            obj_c_hat = problem.get_objective(ch_i, solpool, params_i)
            losses_per_instance.append((obj_c - obj_c_hat).square().mean())
        loss = torch.stack(losses_per_instance)
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss


class pairwiseLTR(optModel):
    """
    Reference:
    """

    def __init__(self, ptoSolver, **kwargs):
        """ """
        super().__init__(ptoSolver)
        # solution pool
        n_vars = ptoSolver.num_vars
        self.solpool = np.empty((0, n_vars), dtype=np.float32)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux,
                ptoSolver=self.ptoSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # solve
        sol_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, self.ptoSolver, **problem.init_API()
        )
        # add into solpool
        self.solpool = np.concatenate((self.solpool, sol_hat))
        # remove duplicate
        self.solpool = np.unique(self.solpool, axis=0)
        solpool = to_tensor(self.solpool).to(coeff_hat.device)
        K = solpool.shape[0]
        # Compute per-instance pairwise loss (supports any batch size).
        loss = []
        for i in range(coeff_hat.shape[0]):
            ch_i = coeff_hat[i:i+1].expand(K, *coeff_hat.shape[1:])
            ct_i = coeff_true[i:i+1].expand(K, *coeff_true.shape[1:])
            params_i = params[i:i+1].expand(K, *params.shape[1:]) if isinstance(params, torch.Tensor) else params
            objpool_c_true_i = problem.get_objective(ct_i, solpool, params_i)
            objpool_c_hat_i = problem.get_objective(ch_i, solpool, params_i)
            # identify the best solution under true coeff
            if self.ptoSolver.modelSense == GRB.MINIMIZE:
                best_ind = torch.argmin(objpool_c_true_i)
            elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
                best_ind = torch.argmax(objpool_c_true_i)
            else:
                raise NotImplementedError
            objpool_cp_best = objpool_c_hat_i[best_ind]
            rest_ind = [j for j in range(K) if j != best_ind.item()]
            objpool_cp_rest = objpool_c_hat_i[rest_ind]
            if self.ptoSolver.modelSense == GRB.MINIMIZE:
                loss.append(F.relu(objpool_cp_best - objpool_cp_rest))
            elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
                loss.append(F.relu(objpool_cp_rest - objpool_cp_best))
            else:
                raise NotImplementedError
        loss = torch.stack(loss)
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss


class listwiseLTR(optModel):
    """
    Reference:
    Code from:
    """

    def __init__(self, ptoSolver, tau=1.0, **kwargs):
        """ """
        super().__init__(ptoSolver)

        if tau <= 0:
            raise ValueError("tau is not positive.")
        self.tau = tau
        # solution pool
        n_vars = ptoSolver.num_vars
        self.solpool = np.empty((0, n_vars), dtype=np.float32)

    def forward(self, problem, coeff_hat, coeff_true, params, **hyperparams):
        """
        Forward pass
        """
        # obtain solution cache if empty
        if len(self.solpool) == 0:
            _, Y_train, Y_train_aux = problem.get_train_data()
            self.solpool, _ = problem.get_decision(
                Y_train,
                params=Y_train_aux,
                ptoSolver=self.ptoSolver,
                isTrain=False,
                **problem.init_API(),
            )
        # solve #TODO: if sol pool reasonable?
        sol_hat, _ = problem.get_decision(
            coeff_hat.detach().cpu(), params, self.ptoSolver, **problem.init_API()
        )
        # add into solpool
        self.solpool = np.concatenate((self.solpool, sol_hat))
        # remove duplicate
        self.solpool = np.unique(self.solpool, axis=0)
        # convert tensor
        solpool = to_tensor(self.solpool).to(coeff_hat.device)
        K = solpool.shape[0]
        # Compute per-instance listwise (softmax ranking) loss (supports any batch size).
        losses_per_instance = []
        for i in range(coeff_hat.shape[0]):
            ch_i = coeff_hat[i:i+1].expand(K, *coeff_hat.shape[1:])
            ct_i = coeff_true[i:i+1].expand(K, *coeff_true.shape[1:])
            params_i = params[i:i+1].expand(K, *params.shape[1:]) if isinstance(params, torch.Tensor) else params
            obj_c = problem.get_objective(ct_i, solpool, params_i)
            obj_c_hat = problem.get_objective(ch_i, solpool, params_i)
            if self.ptoSolver.modelSense == GRB.MINIMIZE:
                loss_i = -(
                    F.log_softmax(-obj_c_hat / self.tau, dim=0)
                    * F.softmax(-obj_c / self.tau, dim=0)
                )
            elif self.ptoSolver.modelSense == GRB.MAXIMIZE:
                loss_i = -(F.log_softmax(obj_c_hat, dim=0) * F.softmax(obj_c, dim=0))
            else:
                raise NotImplementedError
            losses_per_instance.append(loss_i.mean())
        loss = torch.stack(losses_per_instance)
        # reduction
        loss = do_reduction(loss, hyperparams["reduction"])
        return loss
