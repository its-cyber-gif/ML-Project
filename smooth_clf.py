"""Smooth failure model: spline logistic regression on bounded histogram-shape features plus
log operating total (same inputs as the smooth cost model behind v33/v34). Scored by
OOF competition cost with the v34 budget model, alone and blended with the tree ensemble."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from cost_linear_shapes import F, Ft
from cv_eval import y, cost, weights
from policy import decide, competition_cost

pos = y == 1; lc = np.log(cost[pos])
tree = np.load("oof/mu_train_shapes.npy")
mu_b = np.load("oof/bb_k3d1_w1.0_train.npy")
PT = (np.load("oof/lgb2_shapes.npy") + np.load("oof/cat2_shapes.npy")) / 2


def score(p):
    best = None
    for al in [0.75, 1, 1.25, 1.5, 2]:
        f, _ = decide(p, tree, lc - tree[pos], al, 1.0, 0.0)
        _, b = decide(p, mu_b, lc - mu_b[pos], al, 1.0, -1e9)
        c = competition_cost(cost, f, np.where(f == 1, b, 0)).sum()
        if best is None or c < best[0]: best = (c, al)
    return best


def make(C, knots):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         SplineTransformer(n_knots=knots, degree=1), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000))


if __name__ == "__main__":
    print("tree ensemble (lgb+cat shapes): AUC %.4f AP %.4f cost %s" % (roc_auc_score(y, PT), average_precision_score(y, PT), score(PT)))
    for C, knots in [(0.01, 3), (0.003, 3), (0.01, 5)]:
        oof = np.zeros(len(y)); tst = np.zeros(len(Ft))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(F, y):
            m = make(C, knots).fit(F.iloc[a], y[a], logisticregression__sample_weight=weights[a])
            oof[b] = m.predict_proba(F.iloc[b])[:, 1]; tst += m.predict_proba(Ft)[:, 1] / 5
        print(f"smooth C{C} knots{knots}: AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f} cost {score(oof)}", flush=True)
        for w in [0.2, 0.35, 0.5]:
            bl = (1 - w) * PT + w * oof
            print(f"    blend w{w}: AUC {roc_auc_score(y, bl):.4f} AP {average_precision_score(y, bl):.4f} cost {score(bl)}", flush=True)
        np.save(f"oof/sclf_C{C}_k{knots}_train.npy", oof); np.save(f"oof/sclf_C{C}_k{knots}_test.npy", tst)
