"""Compare budget models on failed vehicles via CV pinball loss at tau=0.909
(the competition's under/over penalty ratio 0.20/0.22)."""
import numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from fleet_prep import load
from cv_eval import y, cost
TAU = 0.2 / 0.22
_, _, X, _ = load()
pos = np.where(y == 1)[0]; Xp = X.iloc[pos]; C = cost[pos]; lc = np.log(C)
mu = np.load("oof/mu_train.npy")[pos]
def pin(b): return np.mean(0.2 * np.maximum(C - b, 0) + 0.02 * np.maximum(b - C, 0))
print("global quantile budget           %.2f" % pin(np.quantile(C, TAU)))
# current: exp(mu + quantile of OOF residuals)
res = lc - mu
print("current mu + residual quantile   %.2f" % pin(np.exp(mu + np.quantile(res, TAU))))
for s in [0.8, 1.2]:
    print(f"  residual scale {s}              %.2f" % pin(np.exp(mu + s * np.quantile(res, TAU))))
# direct quantile regression on log cost, with OOF
for leaves, n in [(4, 200), (7, 300), (7, 600)]:
    oof = np.zeros(len(pos))
    for seed in [0, 1, 2]:
        for a, b in KFold(5, shuffle=True, random_state=seed).split(Xp):
            m = lgb.LGBMRegressor(objective="quantile", alpha=TAU, n_estimators=n, learning_rate=0.03,
                                  num_leaves=leaves, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.5, random_state=seed, verbosity=-1)
            m.fit(Xp.iloc[a], lc[a]); oof[b] += m.predict(Xp.iloc[b]) / 3
    print(f"LGBM quantile on log cost ({leaves} leaves, {n} trees) %.2f" % pin(np.exp(oof)))
# heteroscedastic: model |residual| and scale
oof_s = np.zeros(len(pos))
for seed in [0, 1, 2]:
    for a, b in KFold(5, shuffle=True, random_state=seed).split(Xp):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=4, min_child_samples=15,
                              random_state=seed, verbosity=-1).fit(Xp.iloc[a], np.abs(res[a]))
        oof_s[b] += m.predict(Xp.iloc[b]) / 3
z = res / np.maximum(oof_s, 0.05)
print("heteroscedastic mu + s(x)*q(z)    %.2f" % pin(np.exp(mu + np.maximum(oof_s, 0.05) * np.quantile(z, TAU))))
