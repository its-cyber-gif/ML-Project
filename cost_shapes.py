"""Repair-cost model with histogram shape features (the top cost drivers by importance).
Same 5-fold x 3-seed CV as oof_models.run_cost; saves oof/mu_train_shapes.npy
(OOF for failed vehicles, fold average elsewhere) and oof/mu_test_shapes_cv.npy."""
import numpy as np, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from hist_features import with_shapes
from cv_eval import y, cost

_, _, X, Xt = with_shapes()
pos = np.where(y == 1)[0]; Xp = X.iloc[pos]; lc = np.log(cost[pos])
oof = np.zeros(len(pos)); trn = np.zeros(len(X)); tst = np.zeros(len(Xt))
for seed in [0, 1, 2]:
    for a, b in KFold(5, shuffle=True, random_state=seed).split(Xp):
        for m in [lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
                                    random_state=seed, verbosity=-1),
                  CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
                                    random_seed=seed, verbose=False, allow_writing_files=False)]:
            m.fit(Xp.iloc[a], lc[a])
            oof[b] += m.predict(Xp.iloc[b]) / 6; trn += m.predict(X) / 30; tst += m.predict(Xt) / 30
trn[pos] = oof
base = np.load("oof/mu_train.npy")[pos]
rm = lambda v: np.sqrt(np.mean((v - lc) ** 2))
print("cost model RMSE: old %.4f (corr %.4f) -> shapes %.4f (corr %.4f)" % (rm(base), np.corrcoef(base, lc)[0, 1], rm(oof), np.corrcoef(oof, lc)[0, 1]))
np.save("oof/mu_train_shapes.npy", trn); np.save("oof/mu_test_shapes_cv.npy", tst)
