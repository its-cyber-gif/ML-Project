"""Pseudo-test validation: reweight training vehicles by the train->test density ratio
(domain classifier odds) and score policies on the reweighted OOF predictions.

Result: reweighted failure rate is 1.9% (EM estimate for test: 3.5%), and the
reweighted cost prefers risk x1.5 while the leaderboard improved steadily up to x6.
The test depot's extra risk is mostly outside the training range, so covariate
reweighting cannot reproduce it; this proxy is not used to pick the risk level.
It does agree with CV that budgets x0.8-1.0 are best.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from fleet_prep import load
from policy import decide, competition_cost
from cv_eval import y, cost, weights

tr, te, X, Xt = load()
A = pd.concat([X, Xt]); d = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                       colsample_bytree=0.5, subsample=0.8, subsample_freq=1, verbosity=-1)
pd_ = cross_val_predict(m, A, d, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
pt = np.clip(pd_[:len(X)], 1e-4, 1 - 1e-4)
r = np.clip(pt / (1 - pt) * len(X) / len(Xt), 0, None); r = np.minimum(r, np.quantile(r, 0.999))
print("reweighted failure rate %.4f" % np.average(y, weights=weights * r))
mu = np.load("oof/mu_train.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
p = np.mean([np.load(f"oof/p_{n}_w.npy") for n in ("lgb", "xgb", "cat")], 0)
for a in [1.5, 2.5, 4, 6, 8]:
    cc = competition_cost(cost, *decide(p, mu, eps, a, 1.0, 0))
    print(f"alpha {a}: train {cc.mean():.4f}  pseudo-test {np.average(cc, weights=r):.4f}")
