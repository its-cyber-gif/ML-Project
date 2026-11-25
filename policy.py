"""Cost-optimal inspect/budget decisions from probabilities and cost predictions.

Cost per vehicle (competition statement):
  not inspected: C (actual repair cost, 0 if healthy)
  inspected:     10 + 0.20 * max(C - b, 0) + 0.02 * max(b - C, 0)

For a vehicle with failure probability p and conditional repair-cost distribution
C | fail ~ exp(mu + eps) (eps = empirical out-of-fold residuals), the expected
inspection cost is minimised by the 0.20/0.22 = 0.909 quantile of the mixture
(0 with prob 1-p, C|fail with prob p). We inspect only when that expected cost is
below the expected cost of not inspecting, p * E[C | fail].
"""
import numpy as np

UNDER, OVER, FIXED = 0.20, 0.02, 10.0
TAU = UNDER / (UNDER + OVER)


def competition_cost(c, flag, budget):
    return np.where(flag == 1,
                    FIXED + UNDER * np.maximum(c - budget, 0) + OVER * np.maximum(budget - c, 0),
                    c)


def decide(p, mu, eps, alpha=1.0, eps_scale=1.0, margin=0.0, budget_mult=1.0, sigma=None):
    """Return flag, budget arrays.

    alpha       multiplies probabilities (calibration / prior adjustment)
    eps_scale   widens or narrows the residual distribution
    margin      extra cost an inspection must save before we flag
    budget_mult scales the optimal budget (kept at 1 unless CV says otherwise)
    sigma       optional per-vehicle residual scale (heteroscedastic cost model);
                then eps should be standardised residuals z = residual / sigma
    """
    sig = np.ones(len(p)) if sigma is None else np.asarray(sigma)
    p = np.clip(alpha * p, 1e-9, 1 - 1e-9)
    e = np.sort(eps * eps_scale)
    # samples of C|fail per vehicle: shape (n, k)
    out_flag = np.zeros(len(p), dtype=int)
    out_b = np.zeros(len(p))
    for s in range(0, len(p), 4000):
        sl = slice(s, s + 4000)
        C = np.exp(mu[sl, None] + sig[sl, None] * e[None, :])
        k = C.shape[1]
        ps = p[sl]
        u = 1 - (1 - TAU) / ps  # required quantile of C|fail
        idx = np.clip(np.ceil(u * k).astype(int) - 1, 0, k - 1)
        b = np.where(u > 0, C[np.arange(C.shape[0]), idx], 0.0) * budget_mult
        loss_fail = (UNDER * np.maximum(C - b[:, None], 0) + OVER * np.maximum(b[:, None] - C, 0)).mean(1)
        insp = FIXED + ps * loss_fail + (1 - ps) * OVER * b
        no_insp = ps * C.mean(1)
        f = (no_insp - insp > margin).astype(int)
        out_flag[sl] = f
        out_b[sl] = np.where(f == 1, b, 0.0)
    return out_flag, out_b
