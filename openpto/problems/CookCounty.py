import os

import numpy as np
import pandas as pd
import torch
from gurobipy import GRB  # pylint: disable=no-name-in-module
from sklearn.preprocessing import StandardScaler

from openpto.method.utils_method import to_tensor
from openpto.problems.PTOProblem import PTOProblem

FEATURE_COLS = [
    "deaths_sp_lag",
    "INTPTLAT", "INTPTLON",
    "svi_theme1", "svi_theme2", "svi_theme3", "svi_theme4", "svi_total_",
    "prev_deaths_05back", "prev_deaths_04back", "prev_deaths_03back",
    "prev_deaths_02back", "prev_deaths_01back",
]
TARGET_COL = "deaths"


def _build_split_tensor(X_df, Y_df, geoid_order, scaler=None):
    """Convert long-format DataFrames to (T, S, F) and (T, S, 1) tensors.

    geoid_order: list of geoids in fixed spatial order (from training set).
    scaler: if provided, transform X features; otherwise fit+return new scaler.
    Returns (X_tensor, Y_tensor, scaler).
    """
    timesteps = sorted(X_df["timestep"].unique())
    T = len(timesteps)
    S = len(geoid_order)
    F = len(FEATURE_COLS)

    X = np.zeros((T, S, F), dtype=np.float32)
    Y = np.zeros((T, S, 1), dtype=np.float32)

    for t_idx, ts in enumerate(timesteps):
        X_t = X_df[X_df["timestep"] == ts].set_index("geoid")
        Y_t = Y_df[Y_df["timestep"] == ts].set_index("geoid")
        for s_idx, gid in enumerate(geoid_order):
            if gid in X_t.index:
                X[t_idx, s_idx, :] = X_t.loc[gid, FEATURE_COLS].values.astype(np.float32)
            if gid in Y_t.index:
                Y[t_idx, s_idx, 0] = float(Y_t.loc[gid, TARGET_COL])

    X = np.nan_to_num(X, nan=0.0)

    # Fit or apply scaler on flattened (T*S, F)
    X_flat = X.reshape(-1, F)
    if scaler is None:
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(T, S, F)

    return torch.from_numpy(X), torch.from_numpy(Y), scaler


class CookCounty(PTOProblem):
    """Cook County top-K mortality problem.

    Each instance is a timestep; the task is to select the K=100 census tracts
    (out of 1328) with the highest predicted deaths.

    Data: 4 train / 1 val / 2 test timesteps. Fixed real-world splits.
    num_train_instances / num_test_instances are accepted but ignored.
    """

    def __init__(
        self,
        budget=100,
        prob_version="real",
        data_dir="./openpto/data/",
        num_train_instances=0,
        num_test_instances=0,
        rand_seed=0,
        val_frac=None,
        **kwargs,
    ):
        super(CookCounty, self).__init__(data_dir)
        self.budget = budget
        self.prob_version = prob_version

        # data_dir already has problem name appended by args pipeline
        # (./openpto/data/cook_county), so use it directly
        self._load_data(data_dir)

    def _load_data(self, data_path):
        X_train = pd.read_csv(os.path.join(data_path, "train_x.csv"))
        Y_train = pd.read_csv(os.path.join(data_path, "train_y.csv"))
        X_val   = pd.read_csv(os.path.join(data_path, "valid_x.csv"))
        Y_val   = pd.read_csv(os.path.join(data_path, "valid_y.csv"))
        X_test  = pd.read_csv(os.path.join(data_path, "test_x.csv"))
        Y_test  = pd.read_csv(os.path.join(data_path, "test_y.csv"))

        # Fix geoid order from training set for all splits
        geoid_order = sorted(X_train["geoid"].unique())
        self.n_locations = len(geoid_order)

        self.Xs_train, self.Ys_train, scaler = _build_split_tensor(
            X_train, Y_train, geoid_order, scaler=None
        )
        self.Xs_val,  self.Ys_val,  _ = _build_split_tensor(
            X_val,  Y_val,  geoid_order, scaler=scaler
        )
        self.Xs_test, self.Ys_test, _ = _build_split_tensor(
            X_test, Y_test, geoid_order, scaler=scaler
        )

    # ---- PTOProblem interface ----

    def get_train_data(self, **kwargs):
        return self.Xs_train, self.Ys_train, self.Ys_train

    def get_val_data(self, **kwargs):
        return self.Xs_val, self.Ys_val, self.Ys_val

    def get_test_data(self, **kwargs):
        return self.Xs_test, self.Ys_test, self.Ys_test

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        assert Y.ndim == 3  # (B, S, 1)
        assert Z.ndim == 2  # (B, S)
        Y = to_tensor(Y)
        Z = to_tensor(Z).to(Y.device)
        return (Z.unsqueeze(-1) * Y).sum(-1).sum(-1)

    def get_decision(self, Y, params, ptoSolver, isTrain=False, **kwargs):
        Y = to_tensor(Y).cpu()
        output_sols = ptoSolver.solve(Y, self.budget)
        output_vals = self.get_objective(Y, output_sols)
        return output_sols, output_vals

    def get_model_shape(self):
        return len(FEATURE_COLS), 1

    def get_output_activation(self):
        return "none"

    def get_twostageloss(self):
        return "mse"

    def init_API(self):
        return {"modelSense": GRB.MAXIMIZE, "n_vars": self.n_locations}
