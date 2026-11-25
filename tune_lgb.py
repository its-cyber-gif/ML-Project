"""Small CV search over LightGBM settings (same folds for every config)."""
import numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from fleet_prep import load
from cv_eval import y, cost, weights, report

_, _, X, _ = load()
mu = np.load("oof/mu_train.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
base = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.5, reg_lambda=5)
grid = {
    "base": {},
    "leaves7": dict(num_leaves=7),
    "leaves31": dict(num_leaves=31, min_child_samples=20),
    "trees1200_lr01": dict(n_estimators=1200, learning_rate=0.01),
    "col0.3": dict(colsample_bytree=0.3),
    "minchild80": dict(min_child_samples=80),
    "lambda20": dict(reg_lambda=20),
    "trees300": dict(n_estimators=300),
    "trees1000": dict(n_estimators=1000),
    "pos_w3": dict(scale_pos_weight=3),
}
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
for name, kw in grid.items():
    oof = np.zeros(len(X))
    for a, b in folds:
        m = lgb.LGBMClassifier(**{**base, **kw}, random_state=0, verbosity=-1)
        m.fit(X.iloc[a], y[a], sample_weight=weights[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    report(name, oof, mu, eps)
