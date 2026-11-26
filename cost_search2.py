"""Second repair-cost search, on top of the shape features (v30). Each variant is scored by
OOF log-RMSE on the 397 failures AND by OOF competition cost with the v29/v30 classifier."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from hist_features import with_shapes
from cv_eval import y, cost
from policy import decide, competition_cost

_, _, X, Xt = with_shapes()
pos = np.where(y == 1)[0]; lc = np.log(cost[pos])
P = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2


def rates(df):
    tot = df["hist_total"].replace(0, np.nan)
    cols = [c for c in df.columns if c.startswith("s") and "_" not in c]
    return pd.DataFrame({f"{c}_rate": df[c] / tot for c in cols}, index=df.index)


def lgbm(s, **k):
    p = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, random_state=s, verbosity=-1); p.update(k)
    return lgb.LGBMRegressor(**p)


def cat(s, **k):
    p = dict(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10, random_seed=s, verbose=False,
             allow_writing_files=False, thread_count=-1); p.update(k)
    return CatBoostRegressor(**p)


def run(name, D, makers):
    Dp = D.iloc[pos]; oof = np.zeros(len(pos)); trn = np.zeros(len(D))
    for seed in [0, 1, 2]:
        for a, b in KFold(5, shuffle=True, random_state=seed).split(Dp):
            for mk in makers:
                m = mk(seed).fit(Dp.iloc[a], lc[a])
                oof[b] += m.predict(Dp.iloc[b]) / (3 * len(makers)); trn += m.predict(D) / (15 * len(makers))
    trn[pos] = oof
    e = lc - oof
    c = min(competition_cost(cost, *decide(P, trn, e, al, 1.0, 0.0)).sum() for al in [1, 1.25, 1.5])
    print(f"{name:34s} RMSE {np.sqrt(np.mean(e ** 2)):.4f}  OOF cost {c:7.0f}", flush=True)
    np.save("oof/cs2_" + "".join(ch if ch.isalnum() else "_" for ch in name) + ".npy", trn)
    return c


shape_cols = [c for c in X.columns if c[:2] in ("A_", "B_", "C_", "D_", "E_") or c.startswith("hist") or c.endswith("_frac")]
XR = pd.concat([X, rates(X)], axis=1)
if __name__ == "__main__": run("base (v30)", X, [lgbm, cat])
if __name__ == "__main__": run("shapes-only features", X[shape_cols + ["n_missing", "n_zero"]], [lgbm, cat])
if __name__ == "__main__": run("cat only", X, [cat])
if __name__ == "__main__": run("cat depth6 1200it", X, [lambda s: cat(s, depth=6, iterations=1200, learning_rate=0.02)])
if __name__ == "__main__": run("lgb+cat slower/more", X, [lambda s: lgbm(s, n_estimators=800, learning_rate=0.01), lambda s: cat(s, iterations=1500, learning_rate=0.015)])
if __name__ == "__main__": run("+ usage rates", XR, [lgbm, cat])
if __name__ == "__main__": run("cat huber", X, [lambda s: cat(s, loss_function="Huber:delta=1.0")])
