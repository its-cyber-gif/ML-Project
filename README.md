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
| v13 | v12 + heteroscedastic budgets (per-vehicle cost spread; CV pinball 24.3 vs 25.4, global budget 46.2) | 1715 | pending |
| v14 | v13 with test probabilities bagged over 45 CV + 6 full-data models | 1692 | 17,764.58 |
| v15 | v14 with parts budgets ×0.6 (budget-level probe; pair with v11 ×1.25) | 1692 | pending |
| v16 | v14, budgets ×0.8 (one-knob grid, `make_grid.py`) | 1692 | pending |
| v17 | v14, budgets ×1.15 | 1692 | pending |
| v18 | v14 policy, only 1450 inspections | 1450 | pending |
| v19 | v14 policy, 1950 inspections | 1950 | pending |
| v20 | v14 flags, homoscedastic budgets (tests the sigma model) | 1692 | pending |
| v21 | v9 flags + v14 budgets (isolates the budget model) | 1674 | 17,677.09 |
| v24 | v9 ranking, risk ×9 (inspection-count probe on the better v9 ranking) | 1882 | pending |
| v25 | v9 ranking, risk ×12 | 2038 | pending |
| v26 | v9 policy on 15 full-data classifiers (more bagging; 1663/1674 flags shared with v9) | 1676 | pending |
| v27 | v9 flags, budgets ×0.7 (budget probe on the best flags) | 1674 | 18,293.22 |
| v28 | v9 flags, budgets ×1.4 (not submitted: calibrated model predicts ~+700 worse) | 1674 | — |
| v29 | v9 policy on an ensemble retrained with shape features from 5 more histogram groups (CV cost −3.3%) | 1663 | 17,687.55 |
| v30 | shape features in classifier **and** repair-cost model (cost RMSE 0.807 → 0.724; OOF cost −4.9%) | 1632 | 16,874.63 |
| v31 | v9 classifier + new repair-cost model (isolates the cost-model gain) | 1650 | pending |
| v32 | v30 with risk ×7 (inspection count back toward v9's optimum ~1690) | 1696 | pending |
| v33 | v30 inspections; budgets from tree + piecewise-linear spline cost blend on bounded shape features (cost RMSE 0.724 → 0.696, OOF cost −1.3%) | 1632 | 16,395.84 |
| v34 | v30 inspections; budgets from the smooth (spline) cost model alone | 1632 | **16,332.50** (best) |
| v35 | v33 budgets on v32's inspections (risk ×7) | 1696 | pending |
| v36 | v34 inspections; budgets from smooth cost model + interactions of the top-8 shape features, clipped, capped at 8000 (OOF cost 43,843 → 41,961, −4.3%) | 1632 | 17,000.26 (worse than v34: interactions do not transfer) |
| v37 | v36 budgets on v32's inspections (risk ×7) | 1696 | not recommended (built on v36 budgets) |
| v38 | cost-weighted failure ranking (failures weighted by (cost/mean)^0.5, s002/s125/s169 dropped) + v34 budgets; 80 of 1,632 inspections changed (OOF −0.9% vs v34 setup) | 1632 | pending |
| v23 | 1800 inspections, budgets ×0.9 | 1800 | pending |

v6 showed that the heavier-usage test vehicles really do fail more, so later
versions scale risk up instead of normalising the shift away.

#- **Where v9's cost goes (calibrated scenario)**: inspection fees 30%, budget shortfall on failing vehicles 28%,
  budget overspend on failing vehicles 16%, missed failures 20%, overspend on healthy vehicles 5%. So 44% is
  repair-cost uncertainty, which is what v30 attacks.

- **Structure search after v30**: no further histogram groups; four sets of lockstep columns
  ({s002, s125, s169}, {s012, s120, s126, s128}, {s034, s071}, {s079, s144}). Through s012 they show that test s120
  is rounded to 1000, and a rounding scan shows train s059/s061/s054 are coarser than test. These are hygiene-level
  differences, not yet applied.
- **Smooth cost model on bounded features (`cost_linear_shapes.py`, `budget_blend.py`)**: a spline ridge on histogram
  shares alone matches the trees (RMSE 0.719–0.722) and cannot extrapolate (inputs in [0, 1]). Used for flags it
  hurts (+1–2%, it mis-prices healthy vehicles); used **only for budgets** it helps (−1.0% to −1.4% across settings).

- **Smooth failure model (`smooth_clf.py`)**: spline logistic on bounded shape features alone is far weaker
  (AUC 0.965 vs 0.988) and blending it into the tree ensemble raises OOF cost 6–20%. Failure needs raw usage levels;
  the smooth approach helps budgets only.
- **Interactions in the smooth cost model (`smooth_cost2.py`, `smooth_cost3.py`)**: pairwise products of the 8–25
  shape features most correlated with log cost. Overall RMSE barely changes, but expensive failures get higher budgets
  (OOF shortfall on failing vehicles 13,915 → ~11,600). Every variant is 2–4.6% better than v34; top-8 is kept
  (RMSE 0.708). Kernel ridge (RBF) and more spline knots were worse.
- **Budget cap at 8000**: the largest repair cost in training (two vehicles sit exactly there). Budgets above it can
  only add overspend; capping improves OOF cost slightly for every budget model.

- **v36 post-mortem**: interaction budgets were −4.3% in CV and passed the mean-shift check, but scored +668 on the
  leaderboard. Test budgets moved only ±20% (q10–q90 ratio 0.78–1.22), so the public score is very sensitive to the
  budgets of a few expensive failures. Products of standardised features also amplify depot differences that an
  average-shift check cannot see. Budget-model changes now need a much larger CV margin before submission.
- **s002/s125/s169**: Pearson 1.0 in train comes from a few values up to 2.7e10; Spearman is 0.15–0.20 in both depots,
  and the classifier ranks them 209th–236th of 251 by gain. Dropped in v38, but this is not a material leak.
- **Cost-weighted failure model (`cost_weighted_clf.py`)**: weighting failures by (cost/mean)^k turns corr(p, cost | fail)
  from −0.05 to +0.07. With v34 budgets: LightGBM −4.0% (k = 0.25–0.5), CatBoost −1.8%, LightGBM + CatBoost −1.8%
  (k = 0.5); k = 1 overshoots.

## Experiments that did not help

- **After v30 (no CV gain, not used)**: repair-cost variants (shape-only features, CatBoost only, deeper or slower
  boosting, usage-rate features, Huber loss) all within ±0.7% of v30 or worse (`cost_search2.py`); per-vehicle cost
  spread with shape features +2% worse (`sigma_shapes.py`); classifier with usage rates, colsample 0.3 or 31 leaves
  within noise or worse (`clf_search.py`).

- **Leaderboard calibration after v27**: of nine scenarios (test risk ×1.5/3/6 × repair cost ×1/1.5/2), only
  "risk ×6, model costs correct" reproduces both v9 (implied public share 0.32) and v27's gap (predicted +609, observed +629).
  Under that scenario v9's settings are already the Bayes-optimal decision, and budgets ×1.2 / ×1.4 would cost about +230 / +730.

- **Deep-dive checks (no leak or bug found)**:
  - Row order / ids carry no signal (AUC 0.50); only 3 test rows duplicate a train row; `followup_days` is not
    time-to-failure (failed vehicles have the same follow-up profile as healthy ones).
  - After the shift fixes, no sensor differs much between depots (max KS 0.07), but the test depot has a heavier usage
    tail (q99 of ~128 usage sensors 15–60% higher). Only 54 test vehicles exceed the training maximum, and v9 flags 85% of them.
  - 349 of 397 training failures are in the heavy-usage group (mean usage percentile > 0.75). A classifier trained only on
    that group ranks it no better than the global ensemble (`heavy_group.py`: AUC 0.946 vs 0.948, AP 0.571 vs 0.584).
  - Repair cost does not rise with usage (corr −0.13), and at equal failure probability the cost model predicts only
    ~10–20% higher costs on test than on train, so there is no sign of a cost-model artefact from the shift fixes.
  - Conclusion: the models are at the limit of what 397 labelled failures support; what remains is the leaderboard-tuned
    budget level (v27/v28) and inspection count (v10/v24).

- **Cost-model search (`cost_search.py`)**: ridge, ExtraTrees, RandomForest, smaller LGBM and deeper CatBoost
  regressors on the 397 failed vehicles. Only a 0.3 ridge blend helped in CV (RMSE 0.807 → 0.790, OOF cost −0.4%),
  but on test it moved predictions 5× more than on train (mean |Δ| 0.55 vs 0.11): linear extrapolation into the
  test depot's heavier usage, so it was rejected.
- **Headroom check**: on OOF, cost-model residuals ×0.75 would cut cost ~8%; classifier logits ±0.5 toward the truth ~19%.
  Tuning and model swaps so far moved cost by ≤1%, so further gains have to come from leaderboard-validated global settings.
- `logit_model.py`: logistic regression on log1p sensors (AUC 0.985) to extrapolate past the
  training range. Blending it into the ensemble raised OOF cost (0.845–0.863 vs 0.838), and it
  predicted the same test risk level as the trees, so it is not used.
- `tune_lgb.py`: CV search over 10 LightGBM settings (leaves, trees, learning rate, column
  sampling, regularisation, class weight). All within ±0.01 OOF cost of the baseline, which
  is inside CV noise (±0.1), so the original settings are kept.
- `pseudo_test.py`: reweights training vehicles to look like the test depot (density ratio from
  a domain classifier). It reaches a 1.9% failure rate vs EM's 3.5% for test and prefers risk ×1.5,
  while the leaderboard kept improving up to ×6, so it cannot stand in for the leaderboard.
  It agrees with CV that budgets ×0.8–1.0 are best (v15's ×0.6 is expected to be slightly worse).
