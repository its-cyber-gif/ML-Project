"""Per-vehicle spread of the log repair cost, with shape features, on top of the v30 cost
model. Scored by OOF competition cost (v29/v30 classifier). Saves oof/sig2_train.npy,
oof/sig2_test.npy, oof/z2.npy."""
import numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from hist_features import with_shapes
from cv_eval import y, cost
from policy import decide, competition_cost

_, _, X, Xt = with_shapes()
pos = np.where(y == 1)[0]
mu = np.load("oof/mu_train_shapes.npy"); res = np.log(cost[pos]) - mu[pos]
P = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2
Xp = X.iloc[pos]
for leaves, n in [(4, 200), (3, 150)]:
    oof = np.zeros(len(pos)); s_all = np.zeros(len(X)); s_te = np.zeros(len(Xt))
    for seed in [0, 1, 2]:
        for a, b in KFold(5, shuffle=True, random_state=seed).split(Xp):
            m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=leaves, min_child_samples=15,
                                  random_state=seed, verbosity=-1).fit(Xp.iloc[a], np.abs(res[a]))
            oof[b] += m.predict(Xp.iloc[b]) / 3; s_all += m.predict(X) / 15; s_te += m.predict(Xt) / 15
    s_all[pos] = oof; s_all = np.maximum(s_all, 0.05); s_te = np.maximum(s_te, 0.05)
    z = res / s_all[pos]
    for al in [1, 1.25, 1.5]:
        c = competition_cost(cost, *decide(P, mu, z, al, 1.0, 0.0, sigma=s_all)).sum()
        print(f"leaves {leaves}: alpha {al} OOF cost {c:.0f}", flush=True)
    if leaves == 4:
        np.save("oof/sig2_train.npy", s_all); np.save("oof/sig2_test.npy", s_te); np.save("oof/z2.npy", z)
print("homoscedastic reference:", min(competition_cost(cost, *decide(P, mu, res, al, 1.0, 0.0)).sum() for al in [1, 1.25, 1.5]))
