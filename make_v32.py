"""v32: v30 with risk x7 instead of x6. v30 inspects 1632 vehicles (v9: 1674); on v9's
models the leaderboard optimum was ~1690 inspections, so v32 brings v30's count back up."""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost

te = pd.read_csv("data/test.csv", usecols=["id"])
mu = np.load("oof/mu_train_shapes.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
f, b = decide(np.load("oof/full_pt_shapes.npy"), np.load("oof/full_mut_shapes.npy"), eps, 7.0, 1.0, 0.0)
sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
assert len(sub) == 20000 and (sub.loc[sub.flag == 0, "budget"] == 0).all()
sub.to_csv("submission_v32_shapes_risk7.csv", index=False)
f30 = pd.read_csv("submission_v30_shapes_both.csv").flag.to_numpy()
print(f"v32: inspections {f.sum()}, mean budget {b[f == 1].mean():.0f}, v30 flags kept {(f & f30).sum()} of {f30.sum()}")
