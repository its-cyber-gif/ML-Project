# Fleet Maintenance Project — Version History and Improvements

## 1. Project objective

This project predicts which fleet vehicles are likely to fail and estimates the repair budget for each vehicle. The final decision policy balances:

- the expected cost of an undetected failure,
- the inspection fee,
- the repair budget assigned to an inspected vehicle, and
- the cost of over-budgeting or under-budgeting repairs.

The competition metric is total maintenance cost, so **lower public leaderboard scores are better**.

This README documents the development recorded in the supplied version archives, from v5 through v38. Some archives contain a range of versions (for example, `v16-v23`) because they share the same project snapshot. Not every experiment was submitted to Kaggle.

---

## 2. Starting pipeline: v5

The v5 pipeline established the main modelling approach.

### Data preparation
`fleet_prep.py` corrected several train/test sensor-column inconsistencies:

- Histogram bins `s054`–`s063` were shifted by one column in the test data.
- `s120` had an offset of `+16,025,000`.
- `s122` was scaled by `2.54`.
- `s055`, `s057`, and `s062` required rounding.

The README reports that these fixes reduced the train-versus-test classifier AUC from `1.00` to `0.54`, indicating that the most obvious distribution mismatch had largely been removed.

### Predictive models
`oof_models.py` used a 5-fold, 3-seed ensemble of:

- LightGBM,
- XGBoost, and
- CatBoost

to predict failures. Healthy vehicles followed for fewer than 365 days were down-weighted. A separate log repair-cost model was trained on failed vehicles.

### Decision policy
`policy.py` selected a repair budget using the estimated repair-cost distribution (the documented 0.909 quantile) and inspected a vehicle only when the expected inspection-and-repair cost was lower than the expected cost of not inspecting it.

The v5 submission used a risk multiplier of 1.5.

---

## 3. Version-by-version development

The table below reproduces the recorded changes and public leaderboard results. A result marked **pending** was not recorded as a completed public score in the supplied README. A dash means the version was explicitly not submitted or no score was listed.

| Version | Main change or experiment | Inspections | Public score |
|---|---|---:|---:|
| v5 | Corrected test sensor shifts; LightGBM + XGBoost + CatBoost failure ensemble; repair-cost model; expected-cost inspection policy; risk ×1.5 | 1,021 | 22,329.34 |
| v6 | v5 plus quantile-matched test sensors | 754 | 33,860.61 |
| v7 | Returned to the v5 modelling setup and increased risk multiplier to ×2.5 to account for higher test-depot failure risk | 1,265 | 19,770.97 |
| v8 | v5 setup with risk multiplier ×4 | 1,482 | 18,170.52 |
| v9 | v5 setup with risk multiplier ×6 | 1,674 | 17,664.18 |
| v10 | Risk multiplier ×7, selected after fitting a quadratic trend to v7–v9 leaderboard results | 1,754 | Pending |
| v11 | v10 plus parts budgets multiplied by 1.25 | 1,753 | Pending |
| v12 | Added EM prior-shift correction; documented estimated test failure rate of 3.5% (about 4.4× the training rate) and CV-tuned the policy | 1,715 | Pending |
| v13 | v12 plus heteroscedastic repair budgets, allowing predicted cost spread to vary by vehicle | 1,715 | Pending |
| v14 | v13 plus test-probability bagging over 45 CV models and 6 full-data models | 1,692 | 17,764.58 |
| v15 | v14 with parts budgets ×0.6 to probe budget calibration | 1,692 | Pending |
| v16 | v14 with budgets ×0.8; one-factor budget grid experiment | 1,692 | Pending |
| v17 | v14 with budgets ×1.15 | 1,692 | Pending |
| v18 | v14 policy adjusted to make 1,450 inspections | 1,450 | Pending |
| v19 | v14 policy adjusted to make 1,950 inspections | 1,950 | Pending |
| v20 | v14 inspection flags with homoscedastic budgets, testing whether the per-vehicle spread model helped | 1,692 | Pending |
| v21 | Combined v9 inspection flags with v14 budgets to isolate the budget-model contribution | 1,674 | 17,677.09 |
| v22 | No separate v22 result or change was listed in the supplied submission log | — | — |
| v23 | Experiment with 1,800 inspections and budgets ×0.9 | 1,800 | Pending |
| v24 | Used the v9 ranking with risk ×9 to test a higher inspection count | 1,882 | Pending |
| v25 | Used the v9 ranking with risk ×12 | 2,038 | Pending |
| v26 | Applied the v9 policy to 15 full-data classifiers (more bagging); inspection flags were close to v9 | 1,676 | Pending |
| v27 | Kept v9 flags and reduced budgets to ×0.7 | 1,674 | 18,293.22 |
| v28 | Kept v9 flags and increased budgets to ×1.4; README says it was not submitted because calibration predicted a substantially worse score | 1,674 | Not submitted |
| v29 | Retrained the ensemble with shape features from five additional histogram groups; documented a 3.3% CV cost reduction | 1,663 | 17,687.55 |
| v30 | Added shape features to both the failure classifier and repair-cost model; repair-cost RMSE improved from 0.807 to 0.724 and OOF cost improved by 4.9% | 1,632 | 16,874.63 |
| v31 | Kept the v9 classifier and used the new v30 repair-cost model to isolate the cost-model improvement | 1,650 | Pending |
| v32 | Used the v30 model with risk ×7 to bring the inspection count closer to the earlier risk-calibrated range | 1,696 | Pending |
| v33 | Kept v30 inspection decisions; blended tree-based cost predictions with a piecewise-linear spline model for budgets; documented RMSE improvement from 0.724 to 0.696 | 1,632 | 16,395.84 |
| v34 | Kept v30 inspection decisions and used the smooth spline cost model alone for budgets | 1,632 | **16,332.50 — best recorded score** |
| v35 | Applied v33 budgets to the v32 inspection decisions (risk ×7) | 1,696 | Pending |
| v36 | Improved the smooth cost model with interactions among the top eight shape features, clipping and an 8,000 budget cap; CV/OOF cost improved by 4.3% | 1,632 | 17,000.26 |
| v37 | Applied v36 budgets to v32 inspection decisions (risk ×7) | 1,696 | Not recommended in the supplied README |
| v38 | Introduced cost-weighted failure ranking: failure examples weighted by `(cost / mean_cost)^0.5`; dropped `s002`, `s125`, and `s169`; retained v34 budgets. The new ranking changed 80 of 1,632 inspection decisions and showed a 0.9% OOF improvement versus the v34 setup | 1,632 | Pending |

**Important:** The public leaderboard score is not guaranteed to improve monotonically with each version. Several versions were controlled experiments, parameter probes, or ideas that were later rejected.

---

## 4. Main stages of improvement

### Stage A — Correcting data inconsistencies and building the baseline (v5–v6)

The first major improvement was to correct known sensor-column shifts and scale/offset issues in the test data. The baseline then combined three gradient-boosting classifiers and a separate repair-cost model.

The v6 quantile-matching experiment performed much worse on the public leaderboard (33,860.61), so the later versions did not continue with that approach. The project notes instead interpreted the test depot as having a genuinely higher failure risk.

### Stage B — Calibrating inspection decisions (v7–v11)

Rather than changing the underlying v5 ensemble, v7–v9 tested progressively larger risk multipliers:

- v7: ×2.5
- v8: ×4
- v9: ×6

The public score improved at each of these tested settings, from 19,770.97 to 17,664.18. This indicated that inspecting more vehicles was beneficial for this test set. v10 explored ×7, while v11 also changed parts-budget scaling; both were recorded as pending.

### Stage C — Prior-shift correction and model averaging (v12–v14)

v12 introduced an expectation-maximisation (EM) prior-shift adjustment to account for a higher estimated failure prevalence in the test data. v13 added vehicle-specific uncertainty in repair budgets. v14 then bagged test failure probabilities across many cross-validation and full-data models.

v14 achieved a recorded public score of 17,764.58. Although this was not better than v9, it provided a more extensive probability-averaging experiment. v21 later combined v9's inspection decisions with v14's budgets and recorded 17,677.09.

### Stage D — Separating inspection ranking from budget calibration (v15–v28)

This stage tested the two main parts of the policy separately:

1. **Which vehicles should be inspected?** This is driven mainly by the failure-risk ranking and risk multiplier.
2. **How much should be budgeted?** This is driven by the repair-cost model and budget multiplier.

The experiments varied budget multipliers, inspection counts, and model bagging. v27's budget multiplier of 0.7 scored worse than v9 (18,293.22 versus 17,664.18). The v28 ×1.4 budget variant was not submitted because the internal calibration predicted it would be worse.

The notes also report that leaderboard calibration favored the v9 inspection ranking and risk setting among the tested scenarios.

### Stage E — Improving features and repair-cost prediction (v29–v34)

This was the largest modelling improvement after the baseline.

- **v29:** Added shape features from five additional histogram groups. The project notes report a 3.3% CV cost reduction.
- **v30:** Added shape features to both failure classification and repair-cost prediction. Repair-cost RMSE fell from 0.807 to 0.724, with a 4.9% OOF cost improvement.
- **v33:** Tested a blend of tree-based and spline-based repair-cost estimates for budgets. The documented RMSE fell further to 0.696.
- **v34:** Used the smooth spline cost model alone for budgets while retaining the v30 inspection decisions. It achieved the best recorded public score: **16,332.50**.

The project notes found that the smooth model was useful for estimating budgets but not for failure classification. The classifier needed raw usage levels, while the smooth budget model benefited from bounded histogram-shape features.

### Stage F — Testing feature interactions and cost-aware failure ranking (v36–v38)

- **v36:** Added pairwise interactions among the top eight shape features to the smooth repair-cost model, clipped predictions, and capped budgets at 8,000. Despite a 4.3% OOF improvement, its public score was 17,000.26, worse than v34's 16,332.50. The README's post-mortem warns that this interaction improvement did not transfer reliably to the leaderboard.
- **v37:** Reused v36 budgets with a different inspection count. The supplied README does not recommend it.
- **v38:** Tried a cost-weighted failure classifier, giving more weight to expensive failures using `(cost / mean_cost)^0.5`. It also dropped three highly extreme/less useful sensor columns (`s002`, `s125`, `s169`) and retained v34's budget model. Its OOF result was reported as 0.9% better than the v34 setup, but the public leaderboard score was still pending in the supplied version record.

---

## 5. Experiments that were tested but not retained

The version notes document several approaches that were explored and then rejected or not promoted:

- **Quantile-matching the test sensors (v6):** substantially worsened the public score.
- **Logistic regression on log-transformed sensors:** did not improve the tree ensemble's OOF cost.
- **Additional LightGBM hyperparameter tuning:** tested multiple settings, but differences were within the reported cross-validation noise.
- **Pseudo-test reweighting:** estimated a lower test failure rate than the EM adjustment and did not match the leaderboard trend.
- **Alternative repair-cost regressors and feature sets:** most were similar to or worse than the selected model; one ridge blend showed train/test movement that raised extrapolation concerns.
- **Smooth spline failure classifier:** weaker than the tree ensemble and harmful when blended into it.
- **Shape-feature interactions for budgets (v36):** improved OOF cost but worsened the public leaderboard score, demonstrating that CV improvement did not guarantee leaderboard improvement.
- **More spline knots and RBF kernel ridge:** reported as worse than the selected smooth budget approach.

These experiments are included because they explain why the final pipeline kept some methods and discarded others.

---

## 6. Key recorded leaderboard results

The strongest completed public scores listed in the supplied version history are:

| Version | Public score | Notes |
|---|---:|---|
| v34 | **16,332.50** | Best recorded score in the supplied README |
| v33 | 16,395.84 | Tree + spline budget blend |
| v30 | 16,874.63 | Shape features in classifier and repair-cost model |
| v9 | 17,664.18 | Risk ×6 baseline ensemble |
| v21 | 17,677.09 | v9 inspection flags + v14 budgets |
| v29 | 17,687.55 | Additional histogram shape features |
| v14 | 17,764.58 | EM/CV-tuned policy with probability bagging |
| v27 | 18,293.22 | v9 flags with budgets ×0.7 |
| v36 | 17,000.26 | Interaction-based budgets; worse than v34 |

The v38 public score was listed as pending in the supplied archive. Therefore, v34 remains the best **documented completed** public leaderboard result in this version history. This does not establish that v34 is still the best result outside the supplied records.

---

## 7. Reproducibility and interpretation notes

1. **Public score and OOF/CV score are different measurements.** OOF/CV results are internal validation estimates; public leaderboard scores come from the competition's held-out evaluation. They should not be presented as interchangeable.
2. **A version number does not always mean a complete new model.** Some versions only changed a risk multiplier, inspection count, budget multiplier, or one part of the pipeline.
3. **Some version archives are grouped.** The supplied files include combined snapshots such as `v10-v11`, `v16-v23`, `v24-v25`, `v26-v28`, `v30-v31`, `v34-v35`, and `v36-v37`.
4. **The shared `Fleet_Maintenance_Three_Model_Notebook.ipynb` is not the version-specific pipeline notebook.** Each archive also contains a version-named pipeline notebook (for example, `Fleet_v38_pipeline.ipynb`). Use the version-specific notebook when describing or reproducing that version's pipeline.
5. **Pending scores must remain pending.** Do not infer a Kaggle score from an OOF improvement or from a README's prediction.
6. **Best documented version:** v34 has the lowest completed public score in the supplied log (16,332.50). v36 is a useful cautionary experiment: its OOF improvement did not translate into a better public score. v38 is a newer experiment, but its public result is not recorded in the supplied version archive.

---

## 8. Overall conclusion

The project evolved through six broad steps:

1. Fixing known train/test sensor inconsistencies.
2. Building a LightGBM, XGBoost, and CatBoost failure ensemble with a separate repair-cost model.
3. Calibrating inspection risk and the number of vehicles inspected.
4. Testing prior-shift correction, probability bagging, and budget calibration.
5. Improving failure and repair-cost features using histogram-shape information.
6. Testing smooth cost models, feature interactions, and cost-weighted failure ranking.

The strongest documented public leaderboard result is **v34 (16,332.50)**. Later experiments explored whether cost-model interactions and cost-aware failure ranking could improve further, but v36's public result was worse and v38's score was pending in the supplied records.
