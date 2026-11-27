"""Cost-sensitive failure model: failed vehicles weighted by (repair cost / mean)^k, so the
classifier spends its capacity on expensive failures. Scored by OOF competition cost with
the v36 budget model (smooth + top-8 interactions, clipped, capped at 8000).
Also drops s002/s125/s169 (train 'aliases' that are not aliases in test)."""
import sys, numpy as np, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from hist_features import with_shapes
from cv_eval import y, cost, weights
from policy import decide, competition_cost

_, _, X, Xt = with_shapes()
DROP = ["s002", "s125", "s169"]
X, Xt = X.drop(columns=DROP), Xt.drop(columns=DROP)
pos = y == 1; lc = np.log(cost[pos]); LO, HI = np.log(5), np.log(8000)
tree = np.load("oof/mu_train_shapes.npy")
mu_b = np.clip(np.load("oof/sc2_interactions_top8_train.npy"), LO, HI)


def score(p):
    best = None
    for al in [0.5, 0.75, 1, 1.25, 1.5, 2]:
        f, _ = decide(p, tree, lc - tree[pos], al, 1.0, 0.0)
        _, b = decide(p, mu_b, lc - mu_b[pos], al, 1.0, -1e9)
        c = competition_cost(cost, f, np.where(f == 1, np.minimum(b, 8000), 0)).sum()
        if best is None or c < best[0]: best = (round(c), al, int(f.sum()))
    return best


def run(k, models=("lgb", "cat"), seeds=(0, 1)):
    w = weights.copy()
    w[pos] = w[pos] * (cost[pos] / cost[pos].mean()) ** k
    w[pos] *= pos.sum() / w[pos].sum()   # keep total failure weight unchanged
    out = {}
    for name in models:
        oof = np.zeros(len(y))
        for seed in seeds:
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                if name == "lgb":
                    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                                           random_state=seed, verbosity=-1)
                else:
                    m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=8, random_seed=seed,
                                           verbose=False, allow_writing_files=False, thread_count=-1)
                m.fit(X.iloc[a], y[a], sample_weight=w[a]); oof[b] += m.predict_proba(X.iloc[b])[:, 1] / len(seeds)
        out[name] = oof; np.save(f"oof/cw_{name}_k{k}.npy", oof)
    p = np.mean(list(out.values()), 0)
    print(f"k={k} {'+'.join(models)}: AUC {roc_auc_score(y, p):.4f} AP {average_precision_score(y, p):.4f} "
          f"corr(p,cost|fail) {np.corrcoef(p[pos], cost[pos])[0, 1]:+.3f}  OOF cost/alpha/flags {score(p)}", flush=True)
    return p


if __name__ == "__main__":
    ref = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2
    print("reference v34/v36 classifier (lgb+cat shapes, with s002/s125/s169):", score(ref))
    models = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("lgb",)
    for k in [0.0, 0.25, 0.5, 1.0]:
        run(k, models)
