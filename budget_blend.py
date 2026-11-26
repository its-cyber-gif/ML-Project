"""Budgets from a tree + smooth (spline-ridge on bounded shape features) cost blend; flags
keep the v30 tree cost model. Grid over blend weight and spline settings."""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from cost_linear_shapes import F, Ft, pos, lc, tree, tree_t, P
from cv_eval import y, cost
from policy import decide, competition_cost


def spline_cv(knots, deg):
    oof = np.zeros(len(pos)); trn = np.zeros(len(F)); tst = np.zeros(len(Ft))
    for seed in [0, 1, 2]:
        for a, b in KFold(5, shuffle=True, random_state=seed).split(pos):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              SplineTransformer(n_knots=knots, degree=deg), RidgeCV(np.logspace(-1, 4, 25)))
            m.fit(F.iloc[pos[a]], lc[a])
            oof[b] += m.predict(F.iloc[pos[b]]) / 3; trn += m.predict(F) / 15; tst += m.predict(Ft) / 15
    trn[pos] = oof
    return trn, tst


def score(mu_b):
    best = None
    for al in [1, 1.25, 1.5]:
        f, _ = decide(P, tree, lc - tree[pos], al, 1.0, 0.0)
        _, b = decide(P, mu_b, lc - mu_b[pos], al, 1.0, -1e9)
        c = competition_cost(cost, f, np.where(f == 1, b, 0)).sum()
        best = c if best is None or c < best else best
    return best


if __name__ == "__main__":
    for knots, deg in [(4, 2), (5, 3), (3, 1)]:
        trn, tst = spline_cv(knots, deg)
        print(f"spline knots {knots} deg {deg}: alone RMSE {np.sqrt(np.mean((trn[pos] - lc) ** 2)):.4f}")
        for w in [0.5, 0.65, 0.8]:
            mu = (1 - w) * tree + w * trn
            print(f"   w{w}: RMSE {np.sqrt(np.mean((mu[pos] - lc) ** 2)):.4f} OOF cost {score(mu):.0f}", flush=True)
            np.save(f"oof/bb_k{knots}d{deg}_w{w}_train.npy", mu); np.save(f"oof/bb_k{knots}d{deg}_w{w}_test.npy", (1 - w) * tree_t + w * tst)
    print("v30 reference 44517")
