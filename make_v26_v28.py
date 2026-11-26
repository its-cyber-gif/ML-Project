"""v26-v28: refinements of v9, the best leaderboard version (17,664).

Leaderboard so far: v21 (v9 flags + new budget model) 17,677 ~ v9, and v14
(different ranking) 17,765, so v9's ranking is the one to build on.

v26  v9 policy (risk x6) on 15 full-data classifiers instead of 6 (more bagging,
     same settings; 99.97% correlated with v9's probabilities)
v27  v9 flags, budgets x0.7   } budget-level probe on the best flags. Flagged
v28  v9 flags, budgets x1.4   } vehicles are mostly healthy, and each pays 0.02*budget.

Rejected: ridge component in the cost model. CV RMSE improved 0.807 -> 0.790, but on
test it moves predictions 5x more than on train (mean |delta| 0.55 vs 0.11): linear
extrapolation into the test depot's heavier-usage range.
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost

te = pd.read_csv("data/test.csv", usecols=["id"])
mu = np.load("oof/mu_train.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
mut = np.load("oof/full_mut.npy")


def write(name, f, b):
    sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(np.where(f == 1, b, 0), 2)})
    assert len(sub) == 20000 and sub.flag.isin([0, 1]).all() and (sub.budget >= 0).all()
    sub.to_csv(f"submission_{name}.csv", index=False)
    print(f"{name}: inspections {int(f.sum())}, mean budget {sub.budget[sub.flag == 1].mean():.0f}")
    return f


f9, b9 = decide(np.load("oof/full_pt.npy"), mut, eps, 6.0, 1.0, 0.0)
f26 = write("v26_risk6_bag15", *decide(np.load("oof/full_pt5.npy"), mut, eps, 6.0, 1.0, 0.0))
print("v26 flags shared with v9:", int((f26 & f9).sum()), "of", int(f9.sum()))
write("v27_v9_budget070", f9, b9 * 0.7)
write("v28_v9_budget140", f9, b9 * 1.4)
