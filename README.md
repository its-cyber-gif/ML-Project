# ML-Project
Fleet maintenance project

## Improved pipeline (v5)

Cross-validated maintenance cost per vehicle: **0.838** (never inspecting: 3.49).

1. `fleet_prep.py`: fixes the depot shift in `test.csv`. Histogram bins s054–s063 are shifted one column to the right, s120 has a +16,025,000 offset, s122 is scaled by 2.54, and s055/s057/s062 are rounded. After the fixes, a train-vs-test classifier drops from AUC 1.00 to 0.54.
2. `oof_models.py`: 5-fold × 3-seed LightGBM + XGBoost + CatBoost failure models (healthy vehicles followed for less than 365 days are down-weighted) and a log repair-cost model trained on failed vehicles.
3. `policy.py`: for each vehicle, picks the budget that minimises expected cost (the 0.909 quantile of the cost mixture) and inspects only when that is cheaper than not inspecting.
4. `make_submission.py`: writes `submission_v5_ensemble.csv`.

Run it: put the Kaggle CSVs in `data/`, then `python oof_models.py cost w` followed by `python make_submission.py`.
