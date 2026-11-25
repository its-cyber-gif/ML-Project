"""Shared helpers for comparing model versions on out-of-fold (OOF) predictions.

Each version is judged two ways:
  * OOF competition cost on the training depot (policy alpha tuned on OOF)
  * AP / log-loss of the failure probabilities
For test, every version is anchored to the same number of inspections as the
best leaderboard-validated setting (v10: 1754), so a new version changes *which*
vehicles are inspected, not *how many*.
"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from policy import decide, competition_cost

TARGET_INSPECTIONS = 1754

_tr = pd.read_csv("data/train.csv", usecols=["failure", "repair_cost", "followup_days"])
y = _tr.failure.to_numpy()
cost = _tr.repair_cost.fillna(0).to_numpy()
weights = np.where((y == 0) & (_tr.followup_days < 365), _tr.followup_days / 365.0, 1.0)


def oof_cost(p, mu, eps, alphas=(1.0, 1.25, 1.5, 2.0), margins=(0, 1, 2)):
    best = min((competition_cost(cost, *decide(p, mu, eps, a, 1.0, m)).mean(), a, m)
               for a in alphas for m in margins)
    return best


def report(name, p, mu, eps):
    c, a, m = oof_cost(p, mu, eps)
    print(f"{name:28s} AUC {roc_auc_score(y, p):.4f}  AP {average_precision_score(y, p):.4f}  "
          f"LL {log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)):.5f}  OOF cost {c:.4f} (alpha {a}, margin {m})",
          flush=True)
    return c


def test_submission(pt, mut, eps, path, budget_mult=1.0, target=TARGET_INSPECTIONS):
    """Pick alpha so the test inspection count matches the leaderboard-validated target."""
    lo, hi = 0.1, 200.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        n = decide(pt, mut, eps, mid, 1.0, 0.0, budget_mult)[0].sum()
        lo, hi = (mid, hi) if n < target else (lo, mid)
    f, b = decide(pt, mut, eps, hi, 1.0, 0.0, budget_mult)
    te = pd.read_csv("data/test.csv", usecols=["id"])
    sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
    assert len(sub) == 20000 and sub.flag.isin([0, 1]).all() and (sub.budget >= 0).all()
    sub.to_csv(path, index=False)
    print(f"wrote {path}: alpha {hi:.2f}, inspections {int(f.sum())}")
    return sub
