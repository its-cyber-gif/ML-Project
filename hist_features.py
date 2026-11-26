"""Five more 10-bin histogram groups (found by non-negative regression on the s054-s063
total): s023-s032, s037-s046, s080-s089, s129-s138, s154-s163. Each group's bins sum to
a fixed multiple of the same operating total. None is shifted in test (bin shares match).

Trees see raw bin counts, which mostly encode the total (age/usage). Shape features
(bin shares, mean bin, spread, tail share) describe *how* the vehicle is used.
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from fleet_prep import load
from cv_eval import y, cost, weights
from policy import decide, competition_cost

GROUPS = {"A": (23, 32), "B": (37, 46), "C": (80, 89), "D": (129, 138), "E": (154, 163)}


def shape_features(df):
    out = {}
    for k, (a, b) in GROUPS.items():
        h = df[[f"s{i:03d}" for i in range(a, b + 1)]].to_numpy(float)
        tot = np.nansum(h, 1); tot[tot == 0] = np.nan
        fr = h / tot[:, None]
        w = np.arange(10)
        mb = np.nansum(fr * w, 1)
        out.update({f"{k}_f{j}": fr[:, j] for j in range(10)})
        out[f"{k}_mean"] = mb
        out[f"{k}_sd"] = np.sqrt(np.nansum(fr * (w - mb[:, None]) ** 2, 1))
        out[f"{k}_hi"] = np.nansum(fr[:, 7:], 1)
        out[f"{k}_lo"] = np.nansum(fr[:, :2], 1)
    return pd.DataFrame(out, index=df.index)


def with_shapes():
    tr, te, X, Xt = load()
    return tr, te, pd.concat([X, shape_features(tr)], axis=1), pd.concat([Xt, shape_features(te)], axis=1)


if __name__ == "__main__":
    mu = np.load("oof/mu_train.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
    _, _, X0, _ = load(); _, _, X1, _ = with_shapes()
    for name, X in [("baseline", X0), ("+shapes", X1)]:
        oof = np.zeros(len(X))
        for seed in [0, 1]:
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                                       random_state=seed, verbosity=-1).fit(X.iloc[a], y[a], sample_weight=weights[a])
                oof[b] += m.predict_proba(X.iloc[b])[:, 1] / 2
        c = min(competition_cost(cost, *decide(oof, mu, eps, al, 1.0, 0.0)).mean() for al in [1, 1.25, 1.5, 2])
        print(f"{name:9s} AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f} OOF cost {c:.4f}", flush=True)
        np.save(f"oof/lgb2_{name.strip('+')}.npy", oof)
