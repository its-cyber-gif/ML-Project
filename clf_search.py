"""Classifier variants on top of shape features, scored by OOF competition cost with the
v30 cost model (LightGBM, 5-fold x 2 seeds, same splits as hist_features.py)."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from hist_features import with_shapes
from cost_search2 import rates
from cv_eval import y, cost, weights
from policy import decide, competition_cost

_, _, X, _ = with_shapes()
mu = np.load("oof/mu_train_shapes.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
shape_cols = [c for c in X.columns if c[:2] in ("A_", "B_", "C_", "D_", "E_")]


def run(name, D, **k):
    p = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbosity=-1); p.update(k)
    oof = np.zeros(len(D))
    for seed in [0, 1]:
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(D, y):
            m = lgb.LGBMClassifier(random_state=seed, **p).fit(D.iloc[a], y[a], sample_weight=weights[a])
            oof[b] += m.predict_proba(D.iloc[b])[:, 1] / 2
    c = min(competition_cost(cost, *decide(oof, mu, eps, al, 1.0, 0.0)).sum() for al in [1, 1.25, 1.5])
    print(f"{name:28s} AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f} OOF cost {c:7.0f}", flush=True)
    np.save("oof/clf_" + "".join(ch if ch.isalnum() else "_" for ch in name) + ".npy", oof)


run("shapes (v30 lgb)", X)
run("shapes + rates", pd.concat([X, rates(X)], axis=1))
run("shapes, colsample 0.3", X, colsample_bytree=0.3)
run("shapes, 31 leaves", X, num_leaves=31, min_child_samples=60)
