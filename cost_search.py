"""Search for a better conditional repair-cost model (log cost of failed vehicles).
Compares OOF RMSE on the 397 failed vehicles, same 5-fold x 3-seed splits."""
import numpy as np, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from fleet_prep import load
from cv_eval import y, cost

_, _, X, Xt = load()
pos = np.where(y == 1)[0]; Xp = X.iloc[pos].reset_index(drop=True); lc = np.log(cost[pos])
base = np.load("oof/mu_train.npy")[pos]
print("baseline RMSE %.4f corr %.4f" % (np.sqrt(np.mean((base - lc) ** 2)), np.corrcoef(base, lc)[0, 1]))
Xs = np.sign(Xp) * np.log1p(np.abs(Xp))

def cv(make, data, name):
    oof = np.zeros(len(pos))
    for seed in [0, 1, 2]:
        for a, b in KFold(5, shuffle=True, random_state=seed).split(data):
            m = make(seed).fit(data.iloc[a], lc[a]); oof[b] += m.predict(data.iloc[b]) / 3
    print(f"{name:30s} RMSE {np.sqrt(np.mean((oof - lc) ** 2)):.4f} corr {np.corrcoef(oof, lc)[0, 1]:.4f}", flush=True)
    np.save(f"oof/cs_{name}.npy", oof)
    return oof

o = {}
o["ridge"] = cv(lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(np.logspace(-1, 4, 20))), Xs, "ridge")
o["et"] = cv(lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s)), Xp, "et")
o["rf"] = cv(lambda s: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s)), Xp, "rf")
o["lgb_small"] = cv(lambda s: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.01, num_leaves=4, min_child_samples=8, subsample=0.7, subsample_freq=1, colsample_bytree=0.3, reg_lambda=20, random_state=s, verbosity=-1), Xp, "lgb_small")
o["cat_d6"] = cv(lambda s: CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=20, random_seed=s, verbose=False, allow_writing_files=False, thread_count=-1), Xp, "cat_d6")
for k, v in o.items():
    for wgt in [0.3, 0.5]:
        bl = (1 - wgt) * base + wgt * v
        print(f"blend base+{k} w{wgt}: RMSE {np.sqrt(np.mean((bl - lc) ** 2)):.4f}")
