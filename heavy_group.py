"""Where the failures are: ~77% of training failures sit in the heavy-usage group
(mean usage percentile > 0.8, 4.5% of train, 6.8% of test). Does a model trained
only on that group rank vehicles inside it better than the global ensemble?"""
import numpy as np, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from fleet_prep import load
from cv_eval import y, weights

_, _, X, Xt = load()
r = np.load("oof/usage_rank_train.npy")
g = np.where(r > 0.75)[0]
p = np.mean([np.load(f"oof/p_{n}_w.npy") for n in ("lgb", "xgb", "cat")], 0)
print(f"group size {len(g)}, failures {y[g].sum()} of {y.sum()}")
print("global ensemble inside group: AUC %.4f AP %.4f" % (roc_auc_score(y[g], p[g]), average_precision_score(y[g], p[g])))
Xg, yg, wg = X.iloc[g], y[g], weights[g]
oof = np.zeros(len(g))
for seed in [0, 1, 2]:
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xg, yg):
        ms = [lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=10,
                                 random_state=seed, verbosity=-1),
              CatBoostClassifier(iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=10, random_seed=seed,
                                 verbose=False, allow_writing_files=False, thread_count=-1)]
        for m in ms:
            m.fit(Xg.iloc[a], yg[a], sample_weight=wg[a]); oof[b] += m.predict_proba(Xg.iloc[b])[:, 1] / 6
print("group-only model:             AUC %.4f AP %.4f" % (roc_auc_score(yg, oof), average_precision_score(yg, oof)))
from scipy.stats import rankdata
for wt in [0.3, 0.5]:
    bl = (1 - wt) * rankdata(p[g]) + wt * rankdata(oof)
    print(f"rank blend w{wt}:              AUC %.4f AP %.4f" % (roc_auc_score(yg, bl), average_precision_score(yg, bl)))
np.save("oof/group_oof.npy", oof); np.save("oof/group_idx.npy", g)
