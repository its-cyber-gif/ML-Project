"""v33: v30's inspections unchanged; budgets from a blend of the v30 tree cost model and a
piecewise-linear spline ridge on bounded histogram-shape features (budget_blend.py,
knots 3, degree 1, weight 0.5).

CV: cost RMSE 0.724 -> 0.696; OOF competition cost 44,517 -> 43,943 (-1.3%) with flags
unchanged. Bounded inputs (shares in [0, 1]) cannot extrapolate; shift check below.
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost

te = pd.read_csv("data/test.csv", usecols=["id"])
pos = y == 1; lc = np.log(cost[pos])
mu_b = np.load("oof/bb_k3d1_w0.5_train.npy"); mut_b = np.load("oof/bb_k3d1_w0.5_test.npy")
tree = np.load("oof/mu_train_shapes.npy"); tree_t = np.load("oof/full_mut_shapes.npy")
pt = np.load("oof/full_pt_shapes.npy")
P = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2
print("shift check mean |budget-model change|: train p>0.1 %.3f | test p>0.1 %.3f"
      % (np.abs(mu_b - tree)[P > 0.1].mean(), np.abs(mut_b - tree_t)[pt > 0.1].mean()))
f30 = pd.read_csv("submission_v30_shapes_both.csv").flag.to_numpy()
_, b = decide(pt, mut_b, lc - mu_b[pos], 6.0, 1.0, -1e9)
b = np.where(f30 == 1, b, 0.0)
sub = pd.DataFrame({"id": te.id, "flag": f30, "budget": np.round(b, 2)})
assert len(sub) == 20000 and (sub.loc[sub.flag == 0, "budget"] == 0).all() and (sub.budget >= 0).all()
sub.to_csv("submission_v33_budget_blend.csv", index=False)
b30 = pd.read_csv("submission_v30_shapes_both.csv").budget.to_numpy()
print(f"v33: inspections {f30.sum()} (same as v30), mean budget {b[f30 == 1].mean():.0f} (v30 {b30[f30 == 1].mean():.0f}), "
      f"median |budget change| {np.median(np.abs(b - b30)[f30 == 1]):.0f}")
