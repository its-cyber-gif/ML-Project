"""Sharper smooth repair-cost models for budgets (v34 baseline: spline knots 3 degree 1 on
bounded shape features, alone). All inputs stay bounded so nothing can extrapolate."""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from cost_linear_shapes import F, Ft
from cv_eval import y, cost
from policy import decide, competition_cost

pos = np.where(y == 1)[0]; lc = np.log(cost[pos]); posb = y == 1
tree = np.load("oof/mu_train_shapes.npy")
PT = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2
top = None


def score(mu_b):
    best = None
    for al in [1, 1.25, 1.5]:
        f, _ = decide(PT, tree, lc - tree[pos], al, 1.0, 0.0)
        _, b = decide(PT, mu_b, lc - mu_b[pos], al, 1.0, -1e9)
        c = competition_cost(cost, f, np.where(f == 1, b, 0)).sum()
        best = c if best is None or c < best else best
    return best


def cv(name, make, D=F, Dt=Ft):
    oof = np.zeros(len(pos)); trn = np.zeros(len(D)); tst = np.zeros(len(Dt))
    for seed in [0, 1, 2]:
        for a, b in KFold(5, shuffle=True, random_state=seed).split(pos):
            m = make().fit(D.iloc[pos[a]], lc[a])
            oof[b] += m.predict(D.iloc[pos[b]]) / 3; trn += m.predict(D) / 15; tst += m.predict(Dt) / 15
    trn[pos] = oof
    print(f"{name:34s} RMSE {np.sqrt(np.mean((oof - lc) ** 2)):.4f}  OOF cost {score(trn):.0f}", flush=True)
    np.save("oof/sc2_" + "".join(ch if ch.isalnum() else "_" for ch in name) + "_train.npy", trn)
    np.save("oof/sc2_" + "".join(ch if ch.isalnum() else "_" for ch in name) + "_test.npy", tst)
    return trn, tst


pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
cv("v34 spline k3 d1", lambda: make_pipeline(*pre(), SplineTransformer(n_knots=3, degree=1), RidgeCV(np.logspace(-1, 4, 25))))
cv("spline k2 d1 (linear)", lambda: make_pipeline(*pre(), SplineTransformer(n_knots=2, degree=1), RidgeCV(np.logspace(-1, 4, 25))))
cv("spline k4 d1", lambda: make_pipeline(*pre(), SplineTransformer(n_knots=4, degree=1), RidgeCV(np.logspace(-1, 4, 25))))
# pick top-15 bounded features by abs corr with log cost, add pairwise interactions
corr = pd.Series({c: abs(np.corrcoef(F.iloc[pos][c].fillna(F[c].median()), lc)[0, 1]) for c in F.columns}).sort_values(ascending=False)
top = list(corr.index[:15]); print("top features:", top[:8])
from sklearn.compose import ColumnTransformer
inter = lambda: make_pipeline(*pre(), ColumnTransformer([("spl", SplineTransformer(n_knots=3, degree=1), list(range(F.shape[1]))),
                                                        ("int", PolynomialFeatures(2, interaction_only=True, include_bias=False), [F.columns.get_loc(c) for c in top])]),
                              RidgeCV(np.logspace(-1, 4, 25)))
cv("spline k3 d1 + top15 interactions", inter)
for g in [0.003, 0.01]:
    cv(f"kernel ridge rbf g{g}", lambda g=g: make_pipeline(*pre(), KernelRidge(alpha=1.0, kernel="rbf", gamma=g)))
