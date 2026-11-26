"""More full-data bagging for the v9 ranking (v9 used seeds 0-1 of full_fit.py),
plus a ridge component in the repair-cost model (CV RMSE 0.807 -> 0.790).

Saves oof/full_pt5.npy (mean over seeds 0-4, 15 classifiers) and
oof/full_mut_ridge.npy (0.7 * tree cost model + 0.3 * ridge on signed-log features).
"""
import numpy as np, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from fleet_prep import load

tr, te, X, Xt = load()
y = tr.failure.to_numpy(); cost = tr.repair_cost.fillna(0).to_numpy()
w = np.where((y == 0) & (tr.followup_days < 365), tr.followup_days / 365.0, 1.0)
P = []
for seed in [2, 3, 4]:
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
pt5 = (2 * np.load("oof/full_pt.npy") + 3 * np.mean(P, 0)) / 5
np.save("oof/full_pt5.npy", pt5)
print("corr with v9 probabilities %.4f" % np.corrcoef(pt5, np.load("oof/full_pt.npy"))[0, 1])

pos = y == 1
sl = lambda d: np.sign(d) * np.log1p(np.abs(d))
ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(np.logspace(-1, 4, 20)))
ridge.fit(sl(X[pos]), np.log(cost[pos]))
mr = 0.7 * np.load("oof/full_mut.npy") + 0.3 * ridge.predict(sl(Xt))
np.save("oof/full_mut_ridge.npy", mr)
print("cost model change: mean |delta mu| %.3f" % np.abs(mr - np.load("oof/full_mut.npy")).mean())
