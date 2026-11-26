import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from fleet_prep import load
from hist_features import with_shapes
from cv_eval import y, cost, weights
from policy import decide, competition_cost
mu = np.load("oof/mu_train.npy"); eps = np.log(cost[y == 1]) - mu[y == 1]
_, _, X0, _ = load(); _, _, X1, _ = with_shapes()
res = {}
for name, X in [("baseline", X0), ("shapes", X1)]:
    oof = np.zeros(len(X))
    for seed in [0, 1]:
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=8, random_seed=seed,
                                   verbose=False, allow_writing_files=False, thread_count=-1)
            m.fit(X.iloc[a], y[a], sample_weight=weights[a]); oof[b] += m.predict_proba(X.iloc[b])[:, 1] / 2
    res[name] = oof
    c = min(competition_cost(cost, *decide(oof, mu, eps, al, 1.0, 0.0)).mean() for al in [1, 1.25, 1.5, 2])
    print(f"cat {name:9s} AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f} OOF cost {c:.4f}", flush=True)
    np.save(f"oof/cat2_{name}.npy", oof)
for name in ["baseline", "shapes"]:
    lg = np.load(f"oof/lgb2_{name}.npy"); avg = (lg + res[name]) / 2
    c = min(competition_cost(cost, *decide(avg, mu, eps, al, 1.0, 0.0)).mean() for al in [1, 1.25, 1.5, 2])
    print(f"lgb+cat {name:9s} AUC {roc_auc_score(y, avg):.4f} AP {average_precision_score(y, avg):.4f} OOF cost {c:.4f}")
