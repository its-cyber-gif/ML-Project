"""Heteroscedastic repair-cost spread: predict |log-cost residual| per vehicle.

Saves oof/sigma_train.npy (OOF for failed vehicles, fold-average elsewhere),
oof/sigma_test.npy and oof/z.npy (standardised residuals of failed vehicles).
"""
import numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from fleet_prep import load
from cv_eval import y, cost

_, _, X, Xt = load()
pos = np.where(y == 1)[0]
mu = np.load("oof/mu_train.npy")
res = np.log(cost[pos]) - mu[pos]
Xp = X.iloc[pos]
oof = np.zeros(len(pos)); s_all = np.zeros(len(X)); s_test = np.zeros(len(Xt))
SEEDS, NF = [0, 1, 2], 5
for seed in SEEDS:
    for a, b in KFold(NF, shuffle=True, random_state=seed).split(Xp):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=4, min_child_samples=15,
                              random_state=seed, verbosity=-1).fit(Xp.iloc[a], np.abs(res[a]))
        oof[b] += m.predict(Xp.iloc[b]) / len(SEEDS)
        s_all += m.predict(X) / (len(SEEDS) * NF)
        s_test += m.predict(Xt) / (len(SEEDS) * NF)
s_all[pos] = oof
s_all, s_test = np.maximum(s_all, 0.05), np.maximum(s_test, 0.05)
np.save("oof/sigma_train.npy", s_all); np.save("oof/sigma_test.npy", s_test)
np.save("oof/z.npy", res / s_all[pos])
print("sigma train mean %.3f  test mean %.3f" % (s_all.mean(), s_test.mean()))
