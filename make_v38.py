"""v38: cost-weighted failure ranking + v34 budgets.

After v36 (interaction budgets, CV -4.3%) scored 17,000 (+668 vs v34), budgets go back to
v34's smooth model. The change here is WHO gets inspected: the classifier (LightGBM +
CatBoost, as in CV) is trained with failed vehicles weighted by (cost / mean)^0.5, and
s002/s125/s169 are dropped (train correlation 1.0 only through a few huge outliers).

CV with v34 budgets: lgb+cat unweighted 44,245 -> weighted 43,459 (-1.8%); vs the v34
setup 43,843 (-0.9%). Inspections are held at v34's 1,632 so only the ranking changes.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
from hist_features import with_shapes
from policy import decide
from cv_eval import y, cost, weights

K = 0.5
tr, te, X, Xt = with_shapes()
DROP = ["s002", "s125", "s169"]
X, Xt = X.drop(columns=DROP), Xt.drop(columns=DROP)
pos = y == 1; lc = np.log(cost[pos])
w = weights.copy(); w[pos] *= (cost[pos] / cost[pos].mean()) ** K; w[pos] *= pos.sum() / w[pos].sum()
P = []
for seed in [0, 1, 2]:
    for m in [lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                                 random_state=seed, verbosity=-1),
              CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=8, random_seed=seed,
                                 verbose=False, allow_writing_files=False, thread_count=-1)]:
        m.fit(X, y, sample_weight=w); P.append(m.predict_proba(Xt)[:, 1])
    print("seed", seed, "done", flush=True)
pt = np.mean(P, 0); np.save("oof/full_pt_cw05.npy", pt)

tree, tree_t = np.load("oof/mu_train_shapes.npy"), np.load("oof/full_mut_shapes.npy")
mu34, mut34 = np.load("oof/bb_k3d1_w1.0_train.npy"), np.load("oof/bb_k3d1_w1.0_test.npy")
target = int(pd.read_csv("submission_v34_budget_smooth.csv").flag.sum())
lo, hi = 0.5, 50.0
for _ in range(40):
    mid = np.sqrt(lo * hi); n = decide(pt, tree_t, lc - tree[pos], mid, 1.0, 0.0)[0].sum()
    lo, hi = (mid, hi) if n < target else (lo, mid)
f, _ = decide(pt, tree_t, lc - tree[pos], hi, 1.0, 0.0)
_, b = decide(pt, mut34, lc - mu34[pos], hi, 1.0, -1e9)
b = np.where(f == 1, b, 0.0)
sub = pd.DataFrame({"id": te.id, "flag": f.astype(int), "budget": np.round(b, 2)})
assert len(sub) == 20000 and (sub.loc[sub.flag == 0, "budget"] == 0).all() and (sub.budget >= 0).all()
sub.to_csv("submission_v38_costweighted.csv", index=False)
f34 = pd.read_csv("submission_v34_budget_smooth.csv").flag.to_numpy()
print(f"v38: risk x{hi:.2f}, inspections {f.sum()}, mean budget {b[f == 1].mean():.0f}, flags shared with v34 {(f & f34).sum()} of {f34.sum()}")
