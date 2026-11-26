"""Linear repair-cost model on bounded features only (histogram shares, mean bin, spread,
plus log operating total). Bounded inputs cannot extrapolate into the test depot's heavy
tail the way the earlier ridge on raw counters did. Blended with the v30 tree cost model."""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from hist_features import with_shapes
from cv_eval import y, cost
from policy import decide, competition_cost

_, _, X, Xt = with_shapes()
cols = [c for c in X.columns if c[:2] in ("A_", "B_", "C_", "D_", "E_") or c.endswith("_frac") or c == "hist_mean_bin"]


def feats(D):
    F = D[cols].copy(); F["log_total"] = np.log1p(D["hist_total"]); return F


F, Ft = feats(X), feats(Xt)
pos = np.where(y == 1)[0]; lc = np.log(cost[pos])
tree = np.load("oof/mu_train_shapes.npy"); tree_t = np.load("oof/full_mut_shapes.npy")
P = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2
print("bounded features:", F.shape[1], "| log_total train max %.2f test max %.2f" % (F.log_total.max(), Ft.log_total.max()))


def make(kind):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if kind == "spline": steps.append(SplineTransformer(n_knots=4, degree=2))
    return make_pipeline(*steps, RidgeCV(np.logspace(-1, 4, 25)))


for kind in ["linear", "spline"]:
    oof = np.zeros(len(pos)); trn = np.zeros(len(X)); tst = np.zeros(len(Xt))
    for seed in [0, 1, 2]:
        for a, b in KFold(5, shuffle=True, random_state=seed).split(pos):
            m = make(kind).fit(F.iloc[pos[a]], lc[a])
            oof[b] += m.predict(F.iloc[pos[b]]) / 3; trn += m.predict(F) / 15; tst += m.predict(Ft) / 15
    trn[pos] = oof
    print(f"{kind}: alone RMSE {np.sqrt(np.mean((oof - lc) ** 2)):.4f}")
    for w in [0.2, 0.35, 0.5]:
        mu = (1 - w) * tree + w * trn; e = lc - mu[pos]
        c = min(competition_cost(cost, *decide(P, mu, e, al, 1.0, 0.0)).sum() for al in [1, 1.25, 1.5])
        hi = P > 0.1
        d_tr = np.abs(w * (trn - tree))[hi].mean(); d_te = np.abs(w * (tst - tree_t))[np.load("oof/full_pt_shapes.npy") > 0.1].mean()
        print(f"   blend w{w}: RMSE {np.sqrt(np.mean(e ** 2)):.4f} OOF cost {c:7.0f} | shift check |d| train {d_tr:.3f} test {d_te:.3f}")
        np.save(f"oof/mulin_{kind}_w{w}_train.npy", mu); np.save(f"oof/mulin_{kind}_w{w}_test.npy", (1 - w) * tree_t + w * tst)
print("v30 reference OOF cost 44517")
