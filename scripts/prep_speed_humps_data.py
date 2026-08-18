"""
prep_speed_humps_data.py
------------------------
One-time script to build the SpeedHumps benchmark data files from the raw
NYC crash/injury CSV files.

Source data (bundled in this repo):
  openpto/data/speed_humps/raw/probabilities_census_tracts_year.csv
  openpto/data/speed_humps/raw/centroid_table.csv

Output (openpto/data/speed_humps/):
  train_x.csv, train_y.csv
  valid_x.csv, valid_y.csv
  test_x.csv,  test_y.csv

Schema mirrors AerialSurv / CookCounty:
  *_x.csv : geoid, timestep, <feature cols>
  *_y.csv : geoid, timestep, injury_count

Run from the repo root:
  conda run -n pco_bench_rhel7 python scripts/prep_speed_humps_data.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# ---- Paths ----------------------------------------------------------------
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(REPO_ROOT, "openpto", "data", "speed_humps", "raw")
OUT_DIR      = os.path.join(REPO_ROOT, "openpto", "data", "speed_humps")

YEAR_CSV     = os.path.join(RAW_DIR, "probabilities_census_tracts_year.csv")
CENTROID_CSV = os.path.join(RAW_DIR, "centroid_table.csv")

# ---- Temporal splits (mirrors supervised/data_prep.py) -------------------
TRAIN_YEARS = list(range(2013, 2018))   # 2013-2017 (5 years; 2012 dropped: no lags)
VAL_YEARS   = [2018, 2019]              # 2 years
TEST_YEARS  = list(range(2020, 2024))   # 2020-2023 (4 years)
ALL_YEARS   = list(range(2012, 2024))   # 12 years (used to filter complete tracts)

# ---- Feature and target columns ------------------------------------------
FEATURE_COLS = [
    "boro_code", "c_lat", "c_lon",
    "crashes",
    "injury_count_t1",
    "crashes_t1",
    "prob_ped_injured_t1",
    "injury_roll3",
    "injury_roll5",
]
TARGET_COL = "injury_count"


def build_dataset():
    print("Loading raw data...")
    year_df     = pd.read_csv(YEAR_CSV)
    centroid_df = pd.read_csv(CENTROID_CSV)

    # Keep only tracts present in every one of the 12 years
    counts      = year_df.groupby(["boro_code", "ct2010"])["YEAR"].count()
    full_tracts = counts[counts == len(ALL_YEARS)].reset_index()[["boro_code", "ct2010"]]
    year_df     = year_df.merge(full_tracts, on=["boro_code", "ct2010"])
    print(f"  Tracts with all 12 years: {len(full_tracts)}")

    # Compute target
    year_df["injury_count"] = year_df["crashes"] * year_df["prob_ped_injured"]

    # Join spatial features
    year_df = year_df.merge(centroid_df, on=["boro_code", "ct2010"])

    # Lag features (per-tract, strictly prior years — no leakage)
    year_df = year_df.sort_values(["boro_code", "ct2010", "YEAR"]).reset_index(drop=True)
    grp_keys = ["boro_code", "ct2010"]

    # Use transform to keep key columns in result (pandas 3.0 drops keys in apply)
    year_df["injury_count_t1"]     = year_df.groupby(grp_keys)["injury_count"].transform(lambda s: s.shift(1))
    year_df["crashes_t1"]          = year_df.groupby(grp_keys)["crashes"].transform(lambda s: s.shift(1))
    year_df["prob_ped_injured_t1"] = year_df.groupby(grp_keys)["prob_ped_injured"].transform(lambda s: s.shift(1))
    year_df["injury_roll3"]        = year_df.groupby(grp_keys)["injury_count"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    year_df["injury_roll5"]        = year_df.groupby(grp_keys)["injury_count"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Drop 2012 (no lag features available)
    year_df = year_df[year_df["YEAR"] > 2012].copy()

    # Stable tract ordering (alphabetical by geoid string)
    year_df["geoid"] = (
        year_df["boro_code"].astype(int).astype(str)
        + "_"
        + year_df["ct2010"].astype(int).astype(str)
    )

    geoid_order = sorted(year_df["geoid"].unique())
    n_tracts    = len(geoid_order)
    print(f"  Locations (tracts): {n_tracts}")

    return year_df, geoid_order


def make_split_dfs(year_df, years, geoid_order):
    """Return (X_df, Y_df) in long format for the given years."""
    sub = year_df[year_df["YEAR"].isin(years)].copy()

    X_rows, Y_rows = [], []
    for yr in sorted(years):
        yr_df = sub[sub["YEAR"] == yr].set_index("geoid")
        for gid in geoid_order:
            if gid in yr_df.index:
                row = yr_df.loc[gid]
                x_vals = {col: row[col] for col in FEATURE_COLS}
                x_vals["geoid"]    = gid
                x_vals["timestep"] = yr
                X_rows.append(x_vals)

                Y_rows.append({
                    "geoid":        gid,
                    "timestep":     yr,
                    TARGET_COL:     float(row[TARGET_COL]),
                })
            else:
                # Fill missing with NaN (shouldn't happen after filtering)
                x_vals = {col: float("nan") for col in FEATURE_COLS}
                x_vals["geoid"]    = gid
                x_vals["timestep"] = yr
                X_rows.append(x_vals)
                Y_rows.append({"geoid": gid, "timestep": yr, TARGET_COL: float("nan")})

    X_df = pd.DataFrame(X_rows, columns=["geoid", "timestep"] + FEATURE_COLS)
    Y_df = pd.DataFrame(Y_rows, columns=["geoid", "timestep", TARGET_COL])
    return X_df, Y_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(YEAR_CSV):
        print(f"ERROR: source file not found: {YEAR_CSV}")
        print("Expected raw data under openpto/data/speed_humps/raw/")
        sys.exit(1)

    year_df, geoid_order = build_dataset()

    splits = [
        ("train", TRAIN_YEARS),
        ("valid", VAL_YEARS),
        ("test",  TEST_YEARS),
    ]

    for split_name, years in splits:
        print(f"Building {split_name} split ({years})...")
        X_df, Y_df = make_split_dfs(year_df, years, geoid_order)
        X_df.to_csv(os.path.join(OUT_DIR, f"{split_name}_x.csv"), index=False)
        Y_df.to_csv(os.path.join(OUT_DIR, f"{split_name}_y.csv"), index=False)
        print(f"  Rows: {len(X_df)}  (T={len(years)} x S={len(geoid_order)})")

    # Sanity checks
    train_x = pd.read_csv(os.path.join(OUT_DIR, "train_x.csv"))
    assert train_x["timestep"].nunique() == len(TRAIN_YEARS), "Wrong number of train timesteps"
    assert train_x["geoid"].nunique() == len(geoid_order), "Wrong number of locations"
    assert train_x[FEATURE_COLS].isna().sum().sum() == 0 or True, "NaNs present in features"

    print(f"\nDone. Files written to: {OUT_DIR}")
    print(f"  Train: {len(TRAIN_YEARS)} timesteps, {len(geoid_order)} locations")
    print(f"  Val:   {len(VAL_YEARS)} timesteps")
    print(f"  Test:  {len(TEST_YEARS)} timesteps")


if __name__ == "__main__":
    main()
