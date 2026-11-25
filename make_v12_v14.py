"""v12-v14: label-free prior-shift correction + better budgets + more bagging.

v12  EM prior-shift (Saerens et al., 2002): estimate the test depot's failure rate
     from unlabelled test predictions and adjust probabilities in odds space.
     Policy settings (alpha 1.5) are the ones tuned on training OOF cost; no
     leaderboard tuning. EM says the test failure rate is ~4.4x training.
v13  v12 inspections, budgets from a heteroscedastic cost model (per-vehicle
     spread of log repair cost, sigma_model.py). Same flags, -0.8..-2.1% budget
     cost on inspected vehicles in CV.
v14  v13 with test probabilities averaged over the 15 CV fold models per
     algorithm plus the 6 full-data models (more bagging, lower variance).
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost, weights

ALPHA, MARGIN = 1.5, 0.0
te = pd.read_csv("data/test.csv", usecols=["id"])
mu = np.load("oof/mu_train.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
mut = np.load("oof/full_mut.npy")
sig_t = np.load("oof/sigma_test.npy"); z = np.load("oof/z.npy")
ps = np.average(y, weights=weights)


def em_adjust(pt, it=500):
    pi = ps
    for _ in range(it):
        r = (pi / ps) * pt
        q = r / (r + ((1 - pi) / (1 - ps)) * (1 - pt))
        pi = q.mean()
    return q, pi


def write(name, flags, budgets):
    sub = pd.DataFrame({"id": te.id, "flag": flags.astype(int), "budget": np.round(budgets, 2)})
    assert len(sub) == 20000 and sub.flag.isin([0, 1]).all() and (sub.budget >= 0).all()
    assert (sub.loc[sub.flag == 0, "budget"] == 0).all()
    sub.to_csv(f"submission_{name}.csv", index=False)
    print(f"{name}: inspections {int(flags.sum())}, mean budget {budgets[flags == 1].mean():.0f}")


def hetero_budgets(q, flags):
    # force a budget for every vehicle, then keep it only where flagged
    _, b = decide(q, mut, z, ALPHA, 1.0, -1e9, sigma=sig_t)
    return np.where(flags == 1, b, 0.0)


# v12
q12, pi = em_adjust(np.load("oof/full_pt.npy"))
print(f"EM test failure rate {pi:.4f} ({pi / ps:.1f}x train)")
f12, b12 = decide(q12, mut, eps, ALPHA, 1.0, MARGIN)
write("v12_em_prior", f12, b12)

# v13
write("v13_em_hetero_budget", f12, hetero_budgets(q12, f12))

# v14
cv_pt = np.mean([np.load(f"oof/pt_{n}_w.npy") for n in ("lgb", "xgb", "cat")], 0)
pt14 = (15 * cv_pt + 2 * np.load("oof/full_pt.npy")) / 17   # 45 CV models + 6 full-data models
q14, pi14 = em_adjust(pt14)
print(f"v14 EM test failure rate {pi14:.4f}")
f14, _ = decide(q14, mut, eps, ALPHA, 1.0, MARGIN)
write("v14_em_hetero_bagged", f14, hetero_budgets(q14, f14))
