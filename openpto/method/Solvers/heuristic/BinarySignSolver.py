#!/usr/bin/env python
# coding: utf-8
"""
Binary sign solver for scalar-cost decisions: z in {0, 1}, pick z=1 iff sign of
predicted cost favours taking the action.

MINIMIZE: z=1 iff Y <= 0
MAXIMIZE: z=1 iff Y >= 0

Used by the pg_misspec task (PG paper Section 4.1).
"""

import numpy as np

from gurobipy import GRB

from openpto.method.Solvers.abcptoSolver import ptoSolver


class BinarySignSolver(ptoSolver):
    """Trivial threshold solver for scalar-cost binary decisions."""

    def __init__(self, modelSense, n_vars, **kwargs):
        super().__init__(modelSense)
        self.n_vars = n_vars

    def solve(self, Y, **kwargs):
        Y = np.asarray(Y)
        if self.modelSense == GRB.MINIMIZE:
            Z = (Y <= 0).astype(np.float32)
        elif self.modelSense == GRB.MAXIMIZE:
            Z = (Y >= 0).astype(np.float32)
        else:
            raise NotImplementedError(f"Unknown modelSense: {self.modelSense}")
        return Z
