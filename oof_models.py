"""Out-of-fold failure probabilities and conditional repair-cost predictions.

Writes oof/*.npy so the decision policy can be tuned without retraining.
"""
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from fleet_prep import load

SEEDS = [0, 1, 2]
NF = 5
os.makedirs("oof", exist_ok=True)
tr, te, X, Xt = load()
y = tr.failure.to_numpy()
cost = tr.repair_cost.fillna(0).to_numpy()
# Censoring: healthy vehicles observed < 365 days are only partly informative.
w = np.where((y == 0) & (tr.followup_days < 365), tr.followup_days / 365.0, 1.0)


def clf_models(seed):
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15,
                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.5, reg_lambda=5, random_state=seed,
                                  verbosity=-1),
        "xgb": xgb.XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=4,
                                 min_child_weight=3, subsample=0.8, colsample_bytree=0.5,
                                 reg_lambda=5, tree_method="hist", random_state=seed,
                                 eval_metric="logloss", n_jobs=-1),
        "cat": CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5,
                                  l2_leaf_reg=8, random_seed=seed, verbose=False,
                                  allow_writing_files=False, thread_count=-1),
    }


def run_clf(use_w):
    tag = "w" if use_w else "nw"
    oof = {k: np.zeros(len(X)) for k in ["lgb", "xgb", "cat"]}
    tst = {k: np.zeros(len(Xt)) for k in oof}
    for seed in SEEDS:
        skf = StratifiedKFold(NF, shuffle=True, random_state=seed)
        for f, (a, b) in enumerate(skf.split(X, y)):
            for name, m in clf_models(seed).items():
                t = time.time()
                m.fit(X.iloc[a], y[a], sample_weight=w[a] if use_w else None)
                oof[name][b] += m.predict_proba(X.iloc[b])[:, 1] / len(SEEDS)
                tst[name] += m.predict_proba(Xt)[:, 1] / (len(SEEDS) * NF)
            print(f"[{tag}] seed {seed} fold {f} done", flush=True)
    for k in oof:
        np.save(f"oof/p_{k}_{tag}.npy", oof[k]); np.save(f"oof/pt_{k}_{tag}.npy", tst[k])
        print(tag, k, "AUC %.4f AP %.4f LL %.5f" % (roc_auc_score(y, oof[k]),
              average_precision_score(y, oof[k]), log_loss(y, oof[k])))
    avg = np.mean([oof[k] for k in oof], 0)
    print(tag, "avg AUC %.4f AP %.4f" % (roc_auc_score(y, avg), average_precision_score(y, avg)))


def run_cost():
    pos = np.where(y == 1)[0]
    Xp, lc = X.iloc[pos], np.log(cost[pos])
    oof = np.zeros(len(pos)); tst_all = np.zeros(len(Xt)); trn_all = np.zeros(len(X))
    for seed in SEEDS:
        for a, b in KFold(NF, shuffle=True, random_state=seed).split(Xp):
            ms = [lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                    min_child_samples=10, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.5, reg_lambda=10, random_state=seed,
                                    verbosity=-1),
                  CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
                                    random_seed=seed, verbose=False, allow_writing_files=False)]
            for m in ms:
                m.fit(Xp.iloc[a], lc[a])
                oof[b] += m.predict(Xp.iloc[b]) / (2 * len(SEEDS))
                tst_all += m.predict(Xt) / (2 * len(SEEDS) * NF)
                trn_all += m.predict(X) / (2 * len(SEEDS) * NF)
    # For non-failed train rows use the average of fold models (they never saw them).
    trn_all[pos] = oof
    np.save("oof/mu_train.npy", trn_all); np.save("oof/mu_test.npy", tst_all)
    r = np.corrcoef(oof, lc)[0, 1]
    print("cost model: corr(oof, log cost)=%.3f, RMSE %.3f vs const %.3f" % (
        r, np.sqrt(np.mean((oof - lc) ** 2)), lc.std()))


if __name__ == "__main__":
    what = sys.argv[1:] or ["cost", "w", "nw"]
    if "cost" in what: run_cost()
    if "w" in what: run_clf(True)
    if "nw" in what: run_clf(False)
