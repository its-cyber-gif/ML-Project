"""v36-v37: budgets from a smooth cost model with pairwise interactions among the 8 bounded
shape features most correlated with log repair cost (smooth_cost3.py), clipped to the
observed cost range [5, 8000]. Inspections unchanged.

Why it helps: the score charges 0.20 per unit of shortfall, so what matters is the
expensive tail. With interactions the model gives expensive failures higher budgets
(OOF shortfall on failing vehicles 13,915 -> ~11,600). CV: OOF cost 43,843 (v34) -> 42,096
(-4.0%); every variant from 8 to 25 interacting features is 2-4.6% better.

v36  v30/v34 inspections (1632)
v37  v32/v35 inspections (risk x7, 1696)
Budgets are capped at 8000, the largest repair cost in training (two vehicles sit exactly there).
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost

te = pd.read_csv("data/test.csv", usecols=["id"])
pos = y == 1; lc = np.log(cost[pos]); LO, HI = np.log(5), np.log(8000)
mu_b = np.clip(np.load("oof/sc2_interactions_top8_train.npy"), LO, HI)
mut_b = np.clip(np.load("oof/sc2_interactions_top8_test.npy"), LO, HI)
pt = np.load("oof/full_pt_shapes.npy")
PT = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2
tree, tree_t = np.load("oof/mu_train_shapes.npy"), np.load("oof/full_mut_shapes.npy")
print("shift check |budget model - tree|: train p>0.1 %.3f | test p>0.1 %.3f"
      % (np.abs(mu_b - tree)[PT > 0.1].mean(), np.abs(mut_b - tree_t)[pt > 0.1].mean()))
b34 = pd.read_csv("submission_v34_budget_smooth.csv").budget.to_numpy()


def write(name, flags, alpha):
    _, b = decide(pt, mut_b, lc - mu_b[pos], alpha, 1.0, -1e9)
    b = np.where(flags == 1, np.minimum(b, 8000.0), 0.0)  # 8000 = largest repair cost seen (a hard cap)
    sub = pd.DataFrame({"id": te.id, "flag": flags.astype(int), "budget": np.round(b, 2)})
    assert len(sub) == 20000 and (sub.loc[sub.flag == 0, "budget"] == 0).all() and (sub.budget >= 0).all()
    sub.to_csv(f"submission_{name}.csv", index=False)
    print(f"{name}: inspections {flags.sum()}, mean budget {b[flags == 1].mean():.0f}, max {b.max():.0f}")


write("v36_budget_interactions", pd.read_csv("submission_v30_shapes_both.csv").flag.to_numpy(), 6.0)
write("v37_interactions_risk7", pd.read_csv("submission_v32_shapes_risk7.csv").flag.to_numpy(), 7.0)
print(f"(v34 mean budget {b34[b34 > 0].mean():.0f})")
