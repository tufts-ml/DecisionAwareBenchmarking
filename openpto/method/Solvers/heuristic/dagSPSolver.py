#!/usr/bin/env python
# coding: utf-8
"""
DAG shortest path solver for directed grid graphs (right + down edges only).

Used for synthetic shortest path problems (SPO+, PG papers).
Edge indexing matches PyEPO convention:
  - First (rows) * (cols-1) entries: horizontal (right) edges, row by row
  - Next (rows-1) * (cols) entries: vertical (down) edges, column by column
"""

import numpy as np

from openpto.method.Solvers.abcptoSolver import ptoSolver


class dagSPSolver(ptoSolver):
    """Shortest path on a directed grid DAG with edge costs."""

    def __init__(self, modelSense, n_vars, grid_size, **kwargs):
        super().__init__(modelSense)
        self.n_vars = n_vars
        self.grid_size = grid_size
        rows = cols = grid_size
        self.n_horiz = rows * (cols - 1)
        self.n_vert = (rows - 1) * cols

    def _edge_index_right(self, i, j):
        """Index of horizontal edge (i,j)->(i,j+1)."""
        cols = self.grid_size
        return i * (cols - 1) + j

    def _edge_index_down(self, i, j):
        """Index of vertical edge (i,j)->(i+1,j)."""
        cols = self.grid_size
        return self.n_horiz + j * (self.grid_size - 1) + i

    def solve(self, edge_costs, **kwargs):
        """
        Find shortest path from (0,0) to (rows-1, cols-1) on directed grid.

        Args:
            edge_costs: 1-D array of length n_vars (edge costs)

        Returns:
            Z: binary array of length n_vars (1.0 for edges on shortest path)
            others: dict with metadata
        """
        rows = cols = self.grid_size
        INF = 1e18

        # dist[i][j] = min cost to reach (i,j) from (0,0)
        dist = np.full((rows, cols), INF)
        dist[0][0] = 0.0
        prev = {}  # (i,j) -> (pi, pj, edge_idx)

        # DP in topological order (row-major)
        for i in range(rows):
            for j in range(cols):
                # Try arriving from the left: (i, j-1) -> (i, j) via right edge
                if j > 0:
                    eidx = self._edge_index_right(i, j - 1)
                    cost = dist[i][j - 1] + edge_costs[eidx]
                    if cost < dist[i][j]:
                        dist[i][j] = cost
                        prev[(i, j)] = (i, j - 1, eidx)
                # Try arriving from above: (i-1, j) -> (i, j) via down edge
                if i > 0:
                    eidx = self._edge_index_down(i - 1, j)
                    cost = dist[i - 1][j] + edge_costs[eidx]
                    if cost < dist[i][j]:
                        dist[i][j] = cost
                        prev[(i, j)] = (i - 1, j, eidx)

        # Backtrack from (rows-1, cols-1) to (0, 0)
        Z = np.zeros(self.n_vars)
        cur = (rows - 1, cols - 1)
        while cur != (0, 0):
            pi, pj, eidx = prev[cur]
            Z[eidx] = 1.0
            cur = (pi, pj)

        others = {"objective": dist[rows - 1][cols - 1]}
        return Z, others
