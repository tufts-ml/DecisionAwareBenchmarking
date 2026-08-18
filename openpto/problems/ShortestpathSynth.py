import numpy as np
import torch

from gurobipy import GRB

from openpto.method.Solvers.heuristic.dagSPSolver import dagSPSolver
from openpto.method.utils_method import to_array, to_device, to_tensor
from openpto.problems.PTOProblem import PTOProblem


class ShortestpathSynth(PTOProblem):
    """
    Synthetic shortest path on a directed 5x5 grid (right + down edges).

    Supports two variants via prob_version:
      - "synth": SPO+ paper (Elmachtoub & Grigas). Polynomial cost function
        with multiplicative noise. 5 features.
      - "planted": PG losses paper (Gupta & Huang). Two planted paths
        (safe + risky) with context-dependent switching. 6 features.
    """

    def __init__(
        self,
        num_train_instances=400,
        num_test_instances=10000,
        val_frac=0.2,
        rand_seed=2023,
        prob_version="synth",
        num_features=5,
        poly_deg=6,
        noise_width=0.3,
        grid_size=5,
        data_dir="./openpto/data/",
        **kwargs,
    ):
        super().__init__()
        self._set_seed(rand_seed)
        self.prob_version = prob_version
        self.grid_size = grid_size
        self.num_features = num_features
        self.poly_deg = poly_deg
        self.noise_width = noise_width

        rows = cols = grid_size
        self.n_horiz = rows * (cols - 1)
        self.n_vert = (rows - 1) * cols
        self.n_edges = self.n_horiz + self.n_vert

        # Generate data
        N = num_train_instances + num_test_instances
        if prob_version == "synth":
            feats, costs = self._gen_synth_data(
                N, num_features, poly_deg, noise_width, rand_seed
            )
        elif prob_version == "planted":
            feats, costs = self._gen_planted_data(
                N, num_features, poly_deg, noise_width, rand_seed
            )
        else:
            raise ValueError(f"Unknown prob_version: {prob_version}")

        # Solve each instance for ground-truth optimal decisions
        solver = dagSPSolver(
            modelSense=GRB.MINIMIZE, n_vars=self.n_edges, grid_size=grid_size
        )
        sols = []
        for i in range(N):
            z, _ = solver.solve(costs[i])
            sols.append(z)
        sols = np.array(sols)

        # Split: [0, n_vals) = val, [n_vals, num_train) = train, [num_train:] = test
        n_vals = int(val_frac * num_train_instances)
        n_trains = num_train_instances - n_vals

        self.Xs_train = torch.FloatTensor(feats[n_vals:num_train_instances])
        self.Ys_train = torch.FloatTensor(costs[n_vals:num_train_instances])
        self.Zs_train = torch.FloatTensor(sols[n_vals:num_train_instances])

        self.Xs_val = torch.FloatTensor(feats[:n_vals])
        self.Ys_val = torch.FloatTensor(costs[:n_vals])
        self.Zs_val = torch.FloatTensor(sols[:n_vals])

        self.Xs_test = torch.FloatTensor(feats[num_train_instances:])
        self.Ys_test = torch.FloatTensor(costs[num_train_instances:])
        self.Zs_test = torch.FloatTensor(sols[num_train_instances:])

        print(
            f"ShortestpathSynth ({prob_version}): "
            f"train={len(self.Xs_train)}, val={len(self.Xs_val)}, "
            f"test={len(self.Xs_test)}, edges={self.n_edges}"
        )

    def _gen_synth_data(self, n, num_features, poly_deg, noise_width, seed):
        """SPO+ paper: polynomial cost with multiplicative noise."""
        rnd = np.random.RandomState(seed)
        p = num_features
        d = self.n_edges

        # Random binary projection matrix (fixed per seed)
        B = rnd.binomial(1, 0.5, (d, p))
        self.B = B

        # Feature vectors
        feats = rnd.normal(0, 1, (n, p))

        # Cost vectors: c = ((X @ B.T / sqrt(p) + 3)^deg + 1) / 3.5^deg
        costs = (feats @ B.T / np.sqrt(p) + 3) ** poly_deg + 1
        costs /= 3.5 ** poly_deg

        # Multiplicative noise: U(1 - noise_width, 1 + noise_width)
        epsilon = rnd.uniform(1 - noise_width, 1 + noise_width, (n, d))
        costs *= epsilon

        return feats, costs

    def _gen_planted_data(self, n, num_features, poly_deg, noise_width, seed):
        """PG paper: planted safe/risky paths with context-dependent switching."""
        rnd = np.random.RandomState(seed)
        d = self.n_edges
        rows = cols = self.grid_size

        # Features: X_{1:5} ~ N(0,I), X_6 ~ U[0,2]
        feats = np.zeros((n, num_features))
        feats[:, :5] = rnd.normal(0, 1, (n, 5))
        feats[:, 5] = rnd.uniform(0, 2, n)

        # Identify safe and risky path edges
        safe_edges = self._get_safe_path_edges()
        risky_edges = self._get_risky_path_edges()
        all_path_edges = set(safe_edges) | set(risky_edges)
        other_edges = [e for e in range(d) if e not in all_path_edges]

        # Random binary matrix for "other" arcs (uses only first 5 features)
        B_other = rnd.binomial(1, 0.5, (len(other_edges), 5))
        self.B_other = B_other

        # Build cost matrix
        costs = np.zeros((n, d))

        # Safe path: constant cost 2
        for e in safe_edges:
            costs[:, e] = 2.0

        # Risky path: 4*x_6 if x_6 <= 0.55, else 2.2
        x6 = feats[:, 5]
        risky_cost = np.where(x6 <= 0.55, 4.0 * x6, 2.2)
        for e in risky_edges:
            costs[:, e] = risky_cost

        # Other arcs: -((X_{1:5} @ B.T / sqrt(5) + 3)^deg + 1) / 3.5^deg + 2.2
        base = (feats[:, :5] @ B_other.T / np.sqrt(5) + 3) ** poly_deg + 1
        base /= 3.5 ** poly_deg
        other_costs = -base + 2.2
        for idx, e in enumerate(other_edges):
            costs[:, e] = other_costs[:, idx]

        # Additive Gaussian noise
        noise = rnd.normal(0, noise_width, (n, d))
        costs += noise

        return feats, costs

    def _get_safe_path_edges(self):
        """Safe path: top row right, then right column down.
        (0,0)->(0,1)->...>(0,4)->(1,4)->(2,4)->(3,4)->(4,4)"""
        cols = self.grid_size
        edges = []
        # Top row: 4 right edges (0,0)->(0,1), ..., (0,3)->(0,4)
        for j in range(cols - 1):
            edges.append(self._edge_index_right(0, j))
        # Right column: 4 down edges (0,4)->(1,4), ..., (3,4)->(4,4)
        for i in range(self.grid_size - 1):
            edges.append(self._edge_index_down(i, cols - 1))
        return edges

    def _get_risky_path_edges(self):
        """Risky path: left column down, then bottom row right.
        (0,0)->(1,0)->...>(4,0)->(4,1)->(4,2)->(4,3)->(4,4)"""
        cols = self.grid_size
        rows = self.grid_size
        edges = []
        # Left column: 4 down edges (0,0)->(1,0), ..., (3,0)->(4,0)
        for i in range(rows - 1):
            edges.append(self._edge_index_down(i, 0))
        # Bottom row: 4 right edges (4,0)->(4,1), ..., (4,3)->(4,4)
        for j in range(cols - 1):
            edges.append(self._edge_index_right(rows - 1, j))
        return edges

    def _edge_index_right(self, i, j):
        """Index of horizontal edge (i,j)->(i,j+1)."""
        return i * (self.grid_size - 1) + j

    def _edge_index_down(self, i, j):
        """Index of vertical edge (i,j)->(i+1,j)."""
        return self.n_horiz + j * (self.grid_size - 1) + i

    # ---- PTOProblem interface ----

    def get_train_data(self, **kwargs):
        return self.Xs_train, self.Ys_train, self.Zs_train

    def get_val_data(self, **kwargs):
        return self.Xs_val, self.Ys_val, self.Zs_val

    def get_test_data(self, **kwargs):
        return self.Xs_test, self.Ys_test, self.Zs_test

    def get_model_shape(self):
        return self.num_features, self.n_edges

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
        sol = []
        for i in range(len(Y)):
            solp, others = ptoSolver.solve(to_array(Y[i]), **kwargs)
            sol.append(solp)
        sol = to_tensor(np.array(sol))
        obj = self.get_objective(Y, sol, kwargs)
        return sol, obj

    def init_API(self):
        return {
            "modelSense": GRB.MINIMIZE,
            "n_vars": self.n_edges,
            "grid_size": self.grid_size,
        }
