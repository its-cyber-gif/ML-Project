"""Write a submission from the full-data ensemble with a given test-depot risk multiplier.
Usage: python make_risk_variant.py <alpha> <name> [budget_mult]"""
import sys
import numpy as np, pandas as pd
from policy import decide

alpha, name = float(sys.argv[1]), sys.argv[2]
tr = pd.read_csv("data/train.csv", usecols=["failure", "repair_cost"])
te = pd.read_csv("data/test.csv", usecols=["id"])
y = tr.failure.to_numpy(); c = tr.repair_cost.fillna(0).to_numpy()
mu = np.load("oof/mu_train.npy"); eps = np.log(c[y == 1]) - mu[y == 1]
bm = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
p = np.load("oof/full_pt.npy")
f, b = decide(p, np.load("oof/full_mut.npy"), eps, alpha, 1.0, 0.0, bm)
sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
assert len(sub) == 20000 and sub.flag.isin([0, 1]).all() and (sub.budget >= 0).all()
sub.to_csv(f"submission_{name}.csv", index=False)
print(name, "alpha", alpha, "budget_mult", bm, "inspections", int(f.sum()))
