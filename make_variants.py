"""Write submission variants from the full-data fit (run full_fit.py first)."""
import numpy as np, pandas as pd
from policy import decide

tr = pd.read_csv("data/train.csv", usecols=["failure", "repair_cost"])
te = pd.read_csv("data/test.csv", usecols=["id"])
y = tr.failure.to_numpy(); c = tr.repair_cost.fillna(0).to_numpy()
mu = np.load("oof/mu_train.npy"); eps = np.log(c[y == 1]) - mu[y == 1]
V = {
    # quantile-matched test sensors, same policy settings as v5
    "v6_qmatch": (np.load("oof/full_ptq.npy"), np.load("oof/full_mutq.npy"), 1.5, 1.0, 2.0),
    # v5 inputs but more inspections (use if v6 scores worse than v5)
    "v7_aggressive": (np.load("oof/full_pt.npy"), np.load("oof/full_mut.npy"), 2.5, 1.0, 0.0),
}
for name, (p, m, a, s, mg) in V.items():
    f, b = decide(p, m, eps, a, s, mg)
    sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
    assert sub.flag.isin([0, 1]).all() and (sub.budget >= 0).all() and len(sub) == 20000
    sub.to_csv(f"submission_{name}.csv", index=False)
    fl = sub.flag == 1
    print(f"{name}: flags={fl.sum()}  sum(10+0.02*budget)={10*fl.sum()+0.02*sub.budget[fl].sum():.0f}")
