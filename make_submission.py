"""Build the final submission from saved ensemble predictions (run oof_models.py first)."""
import numpy as np, pandas as pd
from policy import decide

ALPHA, EPS_SCALE, MARGIN = 1.5, 1.0, 2.0   # chosen on out-of-fold cost (0.838 vs 3.49 do-nothing)
tr = pd.read_csv("data/train.csv", usecols=["failure", "repair_cost"])
te = pd.read_csv("data/test.csv", usecols=["id"])
y = tr.failure.to_numpy(); c = tr.repair_cost.fillna(0).to_numpy()
mu = np.load("oof/mu_train.npy"); mut = np.load("oof/mu_test.npy")
eps = np.log(c[y == 1]) - mu[y == 1]
pt = np.mean([np.load(f"oof/pt_{n}_w.npy") for n in ("lgb", "xgb", "cat")], 0)
f, b = decide(pt, mut, eps, ALPHA, EPS_SCALE, MARGIN)
sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
samp = pd.read_csv("data/sample_submission.csv")
assert list(sub.columns) == list(samp.columns) and (sub.id.values == samp.id.values).all()
assert sub.flag.isin([0, 1]).all() and np.isfinite(sub.budget).all() and (sub.budget >= 0).all()
assert (sub.loc[sub.flag == 0, "budget"] == 0).all()
sub.to_csv("submission_v5_ensemble.csv", index=False)
print("flagged", sub.flag.sum(), "of", len(sub), "| median budget", sub.loc[sub.flag == 1, "budget"].median())
