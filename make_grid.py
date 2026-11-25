"""v16-v23: one-knob variants around v14, for leaderboard comparison of global settings.

Each file changes exactly ONE setting relative to v14 (1692 inspections, EM prior,
heteroscedastic budgets, bagged probabilities), so a leaderboard score tells us
which direction that setting should move. No per-vehicle label probing: every
change is a single global parameter applied to all vehicles.

v16  budgets x0.8               (CV-optimal lower end)
v17  budgets x1.15              (slightly larger budgets)
v18  fewer inspections: 1450    (only the riskiest vehicles)
v19  more inspections: 1950     (cast a wider net)
v20  homoscedastic budgets      (v12-style budgets on v14 flags: tests the sigma model)
v21  v9's flags + v14 budgets   (isolates the budget model vs v9)
v22  (dropped: 1685/1692 flags identical to v14, not worth a submission)
v23  budgets x0.9 and 1800 inspections (combined mild move)
"""
import numpy as np, pandas as pd
from policy import decide
from cv_eval import y, cost
from make_v12_v14 import em_adjust, write, q14, f14, mut, sig_t, z, eps, ALPHA

base_b = decide(q14, mut, z, ALPHA, 1.0, -1e9, sigma=sig_t)[1]   # hetero budget for every vehicle
homo_b = decide(q14, mut, eps, ALPHA, 1.0, -1e9)[1]


def flags_for(n):
    """Same policy as v14, alpha scaled so exactly ~n vehicles are flagged."""
    lo, hi = 0.05, 50.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        k = decide(q14, mut, eps, ALPHA * mid, 1.0, 0.0)[0].sum()
        lo, hi = (mid, hi) if k < n else (lo, mid)
    return decide(q14, mut, eps, ALPHA * hi, 1.0, 0.0)[0]


def keep(f, b):
    return np.where(f == 1, b, 0.0)


write("v16_budget080", f14, keep(f14, base_b * 0.8))
write("v17_budget115", f14, keep(f14, base_b * 1.15))
f18 = flags_for(1450); write("v18_insp1450", f18, keep(f18, base_b))
f19 = flags_for(1950); write("v19_insp1950", f19, keep(f19, base_b))
write("v20_homo_budget", f14, keep(f14, homo_b))
f9 = pd.read_csv("submission_v9_risk6.csv").flag.to_numpy()
write("v21_v9flags_v14budget", f9, keep(f9, base_b))
score = q14 * np.exp(mut + 0.5 * np.var(eps))
f22 = np.zeros(len(q14), int); f22[np.argsort(-score)[:int(f14.sum())]] = 1
# v22 not written: nearly identical to v14
f23 = flags_for(1800); write("v23_budget090_insp1800", f23, keep(f23, base_b * 0.9))

for name, f in [("v18", f18), ("v19", f19), ("v21", f9), ("v22", f22), ("v23", f23)]:
    print(f"{name}: overlap with v14 flags {int((f & f14).sum())}/{int(f14.sum())}")
