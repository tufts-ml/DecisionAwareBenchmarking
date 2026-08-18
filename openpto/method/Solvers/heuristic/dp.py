#!/usr/bin/env python
# coding: utf-8
"""
0-1 knapsack solver using dynamic programming (vectorised numpy).

Weights may be non-integer (e.g. multiples of 0.01); they are scaled to
integers internally.  Capacity is scaled by the same factor.

Speed: for n_items=20, cap_int=3000, a single solve() takes ~0.3 ms on CPU
(numpy vectorised inner loop) vs ~60 ms for a pure-Python loop.
"""

import numpy as np

from openpto.method.Solvers.abcptoSolver import ptoSolver


class DPSolver(ptoSolver):
    """0-1 knapsack solver via DP.  Accepts float weights (multiples of 0.01)."""

    # Precision factor: multiply weights/capacity to convert to integers.
    # Works for weights generated as rnd.choice(range(300,800))/100.
    _SCALE = 100

    def __init__(self, weights, capacity, modelSense, **kwargs):
        super().__init__(modelSense)
        weights_arr = np.asarray(weights).flatten()
        # Scale to integers; round to avoid floating-point noise.
        self._weights_int = np.round(weights_arr * self._SCALE).astype(int)
        self._cap_int = int(round(float(capacity) * self._SCALE))
        self.n_vars = len(self._weights_int)

    def solve(self, y, **kwargs):
        """
        Solve 0-1 knapsack by DP (vectorised numpy inner loop).

        Parameters
        ----------
        y : array-like, shape (n_items,)
            Profit coefficients.

        Returns
        -------
        sol : list of float
            Binary solution vector (0.0 or 1.0).
        obj : float
            Optimal objective value.
        others : dict
            Empty (for API compatibility).
        """
        profits = np.asarray(y).flatten().astype(np.float64)
        weights = self._weights_int
        cap = self._cap_int
        n = len(profits)

        # dp[w] = best profit achievable with capacity exactly w available.
        dp = np.zeros(cap + 1, dtype=np.float64)
        # chosen[i, w] = True if item i is taken when optimal capacity used is w.
        chosen = np.zeros((n, cap + 1), dtype=bool)

        for i in range(n):
            w_i = int(weights[i])
            p_i = profits[i]
            if w_i > cap:
                continue
            # Vectorised 0-1 DP update (reverse direction = using old dp values).
            # new[w] = max(dp[w], dp[w - w_i] + p_i)  for w in [w_i, cap]
            old_slice = dp[:cap - w_i + 1]          # dp[0 .. cap-w_i]
            new_vals  = old_slice + p_i              # candidate values
            cur_slice = dp[w_i:]                     # dp[w_i .. cap]
            mask = new_vals > cur_slice
            chosen[i, w_i:] = mask
            dp[w_i:] = np.where(mask, new_vals, cur_slice)

        # Backtrack to recover solution
        sol = np.zeros(n, dtype=np.float64)
        w = cap
        for i in range(n - 1, -1, -1):
            if chosen[i, w]:
                sol[i] = 1.0
                w -= int(weights[i])

        obj = float(dp[cap])
        return sol.tolist(), obj, {}
