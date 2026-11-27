"""v34-v35, after v33 (16,395.84) beat v30 by 479 points, more than CV predicted (-1.3%):
the smooth cost model generalises to the test depot better than the trees.

v34  v30 inspections; budgets from the spline ridge on bounded shape features ALONE
     (blend weight 1.0; OOF cost 43,843 vs v33's 43,943; shift check train 0.285 / test 0.284)
v35  v33 budgets (weight 0.5) on v32's inspections (risk x7, 1696 vehicles)
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost

te = pd.read_csv("data/test.csv", usecols=["id"])
pos = y == 1; lc = np.log(cost[pos])
pt = np.load("oof/full_pt_shapes.npy")


def write(name, flags, w, alpha):
    mu_b = np.load(f"oof/bb_k3d1_w{w}_train.npy"); mut_b = np.load(f"oof/bb_k3d1_w{w}_test.npy")
    _, b = decide(pt, mut_b, lc - mu_b[pos], alpha, 1.0, -1e9)
    b = np.where(flags == 1, b, 0.0)
    sub = pd.DataFrame({"id": te.id, "flag": flags.astype(int), "budget": np.round(b, 2)})
    assert len(sub) == 20000 and (sub.loc[sub.flag == 0, "budget"] == 0).all() and (sub.budget >= 0).all()
    sub.to_csv(f"submission_{name}.csv", index=False)
    b33 = pd.read_csv("submission_v33_budget_blend.csv").budget.to_numpy()
    print(f"{name}: inspections {flags.sum()}, mean budget {b[flags == 1].mean():.0f}")
    return b


f30 = pd.read_csv("submission_v30_shapes_both.csv").flag.to_numpy()
f32 = pd.read_csv("submission_v32_shapes_risk7.csv").flag.to_numpy()
write("v34_budget_smooth", f30, 1.0, 6.0)
write("v35_v33_risk7", f32, 0.5, 7.0)
