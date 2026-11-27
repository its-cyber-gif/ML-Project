"""Stability of the interaction gain: number of interacting features, clipping to the
observed log-cost range, and blending with the v34 spline."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from smooth_cost2 import cv, score, F, Ft, pos, lc
import smooth_cost2 as sc

corr = pd.Series({c: abs(np.corrcoef(F.iloc[pos][c].fillna(F[c].median()), lc)[0, 1]) for c in F.columns}).sort_values(ascending=False)
LO, HI = np.log(5), np.log(8000)


def inter(k):
    top = [F.columns.get_loc(c) for c in corr.index[:k]]
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 ColumnTransformer([("spl", SplineTransformer(n_knots=3, degree=1), list(range(F.shape[1]))),
                                                    ("int", PolynomialFeatures(2, interaction_only=True, include_bias=False), top)]),
                                 RidgeCV(np.logspace(-1, 4, 25)))


if __name__ == "__main__":
    base = np.load("oof/sc2_v34_spline_k3_d1_train.npy")
    for k in [8, 10, 15, 20, 25]:
        trn, tst = cv(f"interactions top{k}", inter(k))
        c = score(np.clip(trn, LO, HI)); cb = score(np.clip(0.5 * trn + 0.5 * base, LO, HI))
        print(f"     clipped {c:.0f} | 50/50 with v34 spline, clipped {cb:.0f}", flush=True)
