"""Data loading and depot-shift corrections for the Fleet Maintenance competition.

Findings from adversarial validation (train vs test AUC was 1.000 before fixes):
  * s054..s063 look like a 10-bin histogram. In test the bins are shifted one
    column to the right (test s054 is always 0 and test s055..s063 match train
    s054..s062). We shift them back and mark test s063 as missing.
  * s120 has a constant offset of ~16,025,000 in test.
  * s122 is scaled by ~2.54 in test (looks like a unit change).
  * s047 is entirely missing in test, so it is dropped.
"""
import numpy as np
import pandas as pd

HIST = [f"s{i:03d}" for i in range(54, 64)]
S120_OFFSET = 16_025_000.0
S122_SCALE = 2.54


def fix_test(te: pd.DataFrame) -> pd.DataFrame:
    te = te.copy()
    vals = {f"s{i:03d}": te[f"s{i+1:03d}"].to_numpy() for i in range(54, 63)}
    for c, v in vals.items():
        te[c] = v
    te["s063"] = np.nan
    te["s120"] = np.maximum(te["s120"] - S120_OFFSET, 0)
    te["s122"] = te["s122"] / S122_SCALE
    return te


ROUNDING = {"s055": 10, "s062": 10, "s057": 100}  # test stores these coarser


def fix_train(tr: pd.DataFrame) -> pd.DataFrame:
    tr = tr.copy()
    for c, g in ROUNDING.items():
        tr[c] = np.round(tr[c] / g) * g
    return tr


def add_features(df: pd.DataFrame, sensor_cols) -> pd.DataFrame:
    out = df[sensor_cols].copy()
    out["n_missing"] = df[sensor_cols].isna().sum(1)
    out["n_zero"] = (df[sensor_cols] == 0).sum(1)
    h = df[HIST[:-1]]  # s063 is unavailable in test
    tot = h.sum(1, min_count=1)
    out["hist_total"] = tot
    for c in HIST[:-1]:
        out[f"{c}_frac"] = df[c] / tot.replace(0, np.nan)
    w = np.arange(len(HIST) - 1)
    out["hist_mean_bin"] = (h.fillna(0).to_numpy() * w).sum(1) / tot.replace(0, np.nan)
    return out


def load(data_dir="data"):
    tr = fix_train(pd.read_csv(f"{data_dir}/train.csv"))
    te = fix_test(pd.read_csv(f"{data_dir}/test.csv"))
    sensors = [c for c in te.columns if c.startswith("s") and c not in ("s047", "s063")]
    X = add_features(tr, sensors)
    Xt = add_features(te, sensors)
    return tr, te, X, Xt
