"""Logistic regression on log1p sensors. Unlike trees it extrapolates, which matters
because the test depot's usage tail goes 1.3-1.8x beyond the training range."""
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from fleet_prep import load

tr, te, X, Xt = load()
y = tr.failure.to_numpy()
w = np.where((y == 0) & (tr.followup_days < 365), tr.followup_days / 365.0, 1.0)
cols = [c for c in X.columns if not c.endswith("_frac")]
sl = lambda a: np.sign(a) * np.log1p(np.abs(a))

def model(C):
    return make_pipeline(FunctionTransformer(sl), SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(C=C, max_iter=5000))

for C in [0.003, 0.01, 0.03, 0.1]:
    oof = np.zeros(len(X))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = model(C).fit(X[cols].iloc[a], y[a], logisticregression__sample_weight=w[a])
        oof[b] = m.predict_proba(X[cols].iloc[b])[:, 1]
    print(f"C={C}: AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f}")
    np.save(f"oof/p_lr_C{C}.npy", oof)
    m = model(C).fit(X[cols], y, logisticregression__sample_weight=w)
    np.save(f"oof/pt_lr_C{C}.npy", m.predict_proba(Xt[cols])[:, 1])
