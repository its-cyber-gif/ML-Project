# Running the pipeline

Python 3.10. On macOS, LightGBM needs OpenMP first: `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the competition files in `data/` (`train.csv`, `test.csv`). The zip already includes
`data/` and `oof/` (saved model predictions), so the last step below runs on its own in
under a minute.

## Current best: v34 (16,332.50)

```bash
python make_v34_v35.py
```

v36 (interaction budgets) scored worse (17,000). v38 (cost-weighted ranking) is unscored:

```bash
python make_v38.py
```

## Full rebuild from scratch (about 30-40 minutes)

Each step saves arrays to `oof/` that later steps read.

| Step | Command | Produces |
|---|---|---|
| 1 | `python oof_models.py cost w` | v5 cost model + CV classifier predictions (`mu_train`, `p_*_w`) |
| 2 | `python full_fit.py` | full-data classifier / cost model (`full_pt`, `full_mut`) |
| 3 | `python hist_features.py` | LightGBM CV with and without histogram shape features |
| 4 | `python cat_shapes.py` | CatBoost CV with and without shape features |
| 5 | `python full_fit_shapes.py` | full-data ensemble with shape features (`full_pt_shapes`, `full_mut_shapes`) |
| 6 | `python cost_shapes.py` | tree cost model with shape features (`mu_train_shapes`) |
| 7 | `python make_v30.py` | v30 (inspections used by v33-v36) |
| 8 | `python make_v32.py` | v32 (inspections used by v35/v37) |
| 9 | `python smooth_cost2.py` then `python smooth_cost3.py` | smooth budget models (`sc2_*`) |
| 10 | `python make_v36_v37.py` | v36, v37 |

## How the pieces fit

- `fleet_prep.py`: loads data and undoes the test-depot shifts (histogram bin shift,
  s120 offset, s122 scale, rounding).
- `hist_features.py`: the five extra 10-bin histogram groups and their shape features.
- `policy.py`: turns failure probability + repair-cost distribution into the
  cost-optimal inspect / budget decision.
- `cv_eval.py`: labels, repair costs and censoring weights shared by all scripts.
- `README.md`: every version, its leaderboard score, and every experiment that did not help.

## Competition rules to keep in mind

Only the competition files; allowed libraries only (numpy, pandas, scipy, scikit-learn,
statsmodels, imbalanced-learn, xgboost, lightgbm, catboost, matplotlib, seaborn); no deep
learning, no AutoML; predictions must come from these models, not from an LLM.
