"""v30: shape features in BOTH models.

Leaderboard calibration (after v27) puts ~44% of v9's cost in budget error on vehicles
that do fail, i.e. in repair-cost uncertainty. The histogram shape features turn out to
be the strongest repair-cost drivers (B_f1, hist_mean_bin, C_mean, C_f8, ...), but the
cost model never had them.

  cost model RMSE (log, OOF on 397 failures): 0.807 -> 0.724 (corr 0.872 -> 0.898)
  OOF competition cost (LGB+Cat): old/old 46,810; new classifier only -3.3%; new cost only
  -1.5%; both -4.9%.
  Shift check: on comparable vehicles the cost prediction moves the same amount on train
  (p > 0.1: mean |delta| 0.214) and test (v29-flagged: 0.228), so no extrapolation.

v30  classifier and cost model with shape features, v9's risk x6 policy
v31  v9's classifier + new cost model (isolates the cost-model gain)
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost

te = pd.read_csv("data/test.csv", usecols=["id"])
mu_new = np.load("oof/mu_train_shapes.npy"); eps_new = np.log(cost[y == 1]) - mu_new[y == 1]
mut_new = np.load("oof/full_mut_shapes.npy")
f9 = pd.read_csv("submission_v9_risk6.csv").flag.to_numpy()


def write(name, p):
    f, b = decide(p, mut_new, eps_new, 6.0, 1.0, 0.0)
    sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
    assert len(sub) == 20000 and sub.flag.isin([0, 1]).all() and (sub.budget >= 0).all()
    assert (sub.loc[sub.flag == 0, "budget"] == 0).all()
    sub.to_csv(f"submission_{name}.csv", index=False)
    print(f"{name}: inspections {f.sum()}, mean budget {b[f == 1].mean():.0f}, shared with v9 {(f & f9).sum()}")


write("v30_shapes_both", np.load("oof/full_pt_shapes.npy"))
write("v31_v9clf_newcost", np.load("oof/full_pt.npy"))
