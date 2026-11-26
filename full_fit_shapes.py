"""Full-data ensemble (LightGBM + XGBoost + CatBoost, 3 seeds) with histogram shape
features (hist_features.py). Saves oof/full_pt_shapes.npy and oof/full_mut_shapes.npy."""
import numpy as np, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from hist_features import with_shapes

tr, te, X, Xt = with_shapes()
y = tr.failure.to_numpy(); cost = tr.repair_cost.fillna(0).to_numpy()
w = np.where((y == 0) & (tr.followup_days < 365), tr.followup_days / 365.0, 1.0)
P = []
for seed in [0, 1, 2]:
    for m in [lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                                 random_state=seed, verbosity=-1),
              xgb.XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=3,
                                subsample=0.8, colsample_bytree=0.5, reg_lambda=5, tree_method="hist",
                                random_state=seed, n_jobs=-1),
              CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=8,
                                 random_seed=seed, verbose=False, allow_writing_files=False)]:
        m.fit(X, y, sample_weight=w); P.append(m.predict_proba(Xt)[:, 1])
    print("seed", seed, "done", flush=True)
pos = y == 1
MU = []
for seed in [0, 1]:
    for m in [lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
                                random_state=seed, verbosity=-1),
              CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
                                random_seed=seed, verbose=False, allow_writing_files=False)]:
        m.fit(X[pos], np.log(cost[pos])); MU.append(m.predict(Xt))
pt = np.mean(P, 0)
np.save("oof/full_pt_shapes.npy", pt); np.save("oof/full_mut_shapes.npy", np.mean(MU, 0))
old = np.load("oof/full_pt.npy")
print("mean test p %.4f (old %.4f); corr with old %.4f" % (pt.mean(), old.mean(), np.corrcoef(pt, old)[0, 1]))
