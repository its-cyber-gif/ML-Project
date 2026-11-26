"""v29: v9's policy (risk x6, same cost model and budgets rule) on the classifier ensemble
retrained with histogram shape features (hist_features.py).

CV (LightGBM + CatBoost, 5-fold x 2 seeds): OOF cost 0.8359 -> 0.8085 (-3.3%).
Leaderboard calibration after v27 says v9's risk x6 is the right setting for this
probability scale; the new ensemble has the same mean test probability (0.0211 vs 0.0212).
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost

te = pd.read_csv("data/test.csv", usecols=["id"])
mu = np.load("oof/mu_train.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
mut = np.load("oof/full_mut.npy")
f, b = decide(np.load("oof/full_pt_shapes.npy"), mut, eps, 6.0, 1.0, 0.0)
sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
assert len(sub) == 20000 and sub.flag.isin([0, 1]).all() and (sub.budget >= 0).all()
assert (sub.loc[sub.flag == 0, "budget"] == 0).all()
sub.to_csv("submission_v29_shapes.csv", index=False)
f9 = pd.read_csv("submission_v9_risk6.csv").flag.to_numpy()
print(f"v29: inspections {f.sum()}, mean budget {b[f == 1].mean():.0f}, flags shared with v9 {(f & f9).sum()} of {f9.sum()}")
