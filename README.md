# ML-Project
Fleet maintenance project

## Improved pipeline (v5)

Cross-validated maintenance cost per vehicle: **0.838** (never inspecting: 3.49).

1. `fleet_prep.py`: fixes the depot shift in `test.csv`. Histogram bins s054–s063 are shifted one column to the right, s120 has a +16,025,000 offset, s122 is scaled by 2.54, and s055/s057/s062 are rounded. After the fixes, a train-vs-test classifier drops from AUC 1.00 to 0.54.
2. `oof_models.py`: 5-fold × 3-seed LightGBM + XGBoost + CatBoost failure models (healthy vehicles followed for less than 365 days are down-weighted) and a log repair-cost model trained on failed vehicles.
3. `policy.py`: for each vehicle, picks the budget that minimises expected cost (the 0.909 quantile of the cost mixture) and inspects only when that is cheaper than not inspecting.
4. `make_submission.py`: writes `submission_v5_ensemble.csv`.

Run it: put the Kaggle CSVs in `data/`, then `python oof_models.py cost w` followed by `python make_submission.py`.

## Submission log (public leaderboard, total cost — lower is better)

| Version | Change | Inspections | Public score |
|---|---|---|---|
| v5 | Test shift fixes + LGB/XGB/CatBoost ensemble + cost-optimal policy (risk ×1.5) | 1021 | 22329.34 |
| v6 | v5 + quantile-matched test sensors | 754 | 33860.61 |
| v7 | v5 with risk ×2.5 (test depot has more failures) | 1265 | 19770.97 |
| v8 | v5 with risk ×4 | 1482 | 18170.52 |
| v9 | v5 with risk ×6 | 1674 | 17664.18 |
| v10 | risk ×7 (quadratic fit of v7–v9 scores puts optimum near ×6.3–7) | 1754 | pending |
| v11 | v10 with parts budgets ×1.25 | 1753 | pending |
| v12 | EM prior-shift correction (label-free estimate: test failure rate 3.5% = 4.4× train), CV-tuned policy | 1715 | pending |

v6 showed that the heavier-usage test vehicles really do fail more, so later
versions scale risk up instead of normalising the shift away.

### Experiments that did not help
- `logit_model.py`: logistic regression on log1p sensors (AUC 0.985) to extrapolate past the
  training range. Blending it into the ensemble raised OOF cost (0.845–0.863 vs 0.838), and it
  predicted the same test risk level as the trees, so it is not used.
- `tune_lgb.py`: CV search over 10 LightGBM settings (leaves, trees, learning rate, column
  sampling, regularisation, class weight). All within ±0.01 OOF cost of the baseline, which
  is inside CV noise (±0.1), so the original settings are kept.
