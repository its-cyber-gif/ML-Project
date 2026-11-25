"""Fit the ensemble on all training data; predict test both as-is and after
quantile-matching each test sensor to the training distribution (removes the
remaining depot-level scale shift in the heavy-usage tail)."""
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from fleet_prep import load, add_features

def qmatch(train_col, test_col):
    a = np.sort(train_col.dropna().to_numpy()); t = test_col.copy()
    m = t.notna()
    r = t[m].rank(pct=True, method="average").to_numpy()
    t[m] = np.quantile(a, np.clip(r - 0.5 / m.sum(), 0, 1))
    return t

tr, te, X, Xt = load()
sensors = [c for c in Xt.columns if c.startswith("s") and "_" not in c]
te_q = te.copy()
for c in sensors:
    te_q[c] = qmatch(tr[c], te[c])
Xq = add_features(te_q, sensors)
y = tr.failure.to_numpy(); cost = tr.repair_cost.fillna(0).to_numpy()
w = np.where((y == 0) & (tr.followup_days < 365), tr.followup_days / 365.0, 1.0)
P, PQ = [], []
for seed in [0, 1]:
    ms = [lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                             random_state=seed, verbosity=-1),
          xgb.XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=3,
                            subsample=0.8, colsample_bytree=0.5, reg_lambda=5, tree_method="hist",
                            random_state=seed, n_jobs=-1),
          CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=8,
                             random_seed=seed, verbose=False, allow_writing_files=False)]
    for m in ms:
        m.fit(X, y, sample_weight=w)
        P.append(m.predict_proba(Xt)[:, 1]); PQ.append(m.predict_proba(Xq)[:, 1])
pos = y == 1
MU, MUQ = [], []
for seed in [0, 1]:
    for m in [lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
                                random_state=seed, verbosity=-1),
              CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
                                random_seed=seed, verbose=False, allow_writing_files=False)]:
        m.fit(X[pos], np.log(cost[pos])); MU.append(m.predict(Xt)); MUQ.append(m.predict(Xq))
np.save("oof/full_pt.npy", np.mean(P, 0)); np.save("oof/full_ptq.npy", np.mean(PQ, 0))
np.save("oof/full_mut.npy", np.mean(MU, 0)); np.save("oof/full_mutq.npy", np.mean(MUQ, 0))
print("mean test p: as-is %.4f  quantile-matched %.4f  (train rate %.4f)" % (np.mean(P), np.mean(PQ), y.mean()))
