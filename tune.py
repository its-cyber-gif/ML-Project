"""Tune the decision policy on out-of-fold predictions and compare to baselines."""
import itertools, sys
import numpy as np
import pandas as pd
from policy import decide, competition_cost

tr = pd.read_csv("data/train.csv", usecols=["failure", "repair_cost", "followup_days"])
y = tr.failure.to_numpy(); c = tr.repair_cost.fillna(0).to_numpy()
pos = y == 1
mu = np.load("oof/mu_train.npy")
eps = np.log(c[pos]) - mu[pos]


def P(tag, names=("lgb", "xgb", "cat")):
    return np.mean([np.load(f"oof/p_{n}_{tag}.npy") for n in names], 0)


def score(flag, b):
    return competition_cost(c, flag, b).mean()


print("Do nothing (never inspect):          %.4f" % c.mean())
q90 = np.quantile(c[pos], 0.9)
for tag in ["w", "nw"]:
    for names in [("lgb",), ("xgb",), ("cat",), ("lgb", "xgb", "cat")]:
        p = P(tag, names)
        # notebook-style baseline: threshold on p*exp(mu), fixed q90 budget
        e = p * np.exp(mu)
        best = min(score((e > t).astype(int), np.where(e > t, q90, 0))
                   for t in np.quantile(e, np.linspace(0.9, 0.9995, 200)))
        f, b = decide(p, mu, eps)
        print(f"{tag:>2} {'+'.join(names):12s} threshold+q90 best={best:.4f} | "
              f"optimal policy (no tuning)={score(f, b):.4f} flags={f.sum()}")

tag = sys.argv[1] if len(sys.argv) > 1 else "w"
p = P(tag)
rows = []
for a, s, m, bm in itertools.product([0.6, 0.8, 1.0, 1.2, 1.5], [0.8, 1.0, 1.2],
                                     [-2, 0, 2, 5], [0.8, 1.0, 1.2]):
    f, b = decide(p, mu, eps, a, s, m, bm)
    rows.append((a, s, m, bm, score(f, b), f.sum()))
r = pd.DataFrame(rows, columns=["alpha", "eps_scale", "margin", "budget_mult", "cost", "flags"])
print(r.sort_values("cost").head(10).to_string(index=False))
