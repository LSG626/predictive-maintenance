# Engine-Grouped Remaining Useful Life Estimation on NASA C-MAPSS FD001

## Abstract

This study establishes a reproducible baseline for remaining useful life (RUL) estimation on the NASA C-MAPSS FD001 turbofan degradation dataset. Three regression models—ordinary least squares, Ridge regression, and Random Forest—are compared using three-fold cross-validation grouped by engine, so observations from a given engine do not appear in both training and validation folds. The benchmark uses the 24 operating-setting and sensor measurements at each cycle and the conventional training target defined as the engine's maximum observed cycle minus the current cycle. On 20,631 training observations from 100 engines, Random Forest achieved the best pooled out-of-fold RMSE (42.828 cycles), compared with 44.974 for Ridge and 44.983 for ordinary least squares. This is a baseline comparison, not a claim of state-of-the-art performance: only one subset and three models are included, and the official test set is not used for model selection. Code and machine-readable results are provided with the project to support reproduction.

**Keywords:** predictive maintenance, remaining useful life, C-MAPSS, turbofan, engine-grouped validation, regression

## 1. Introduction

Predictive maintenance uses condition-monitoring data to estimate degradation and support maintenance decisions. RUL estimation is a regression formulation in which the model estimates the remaining operating cycles before an end-of-life event. Multivariate sensor trajectories present a validation challenge: rows from one physical asset are correlated, and a random row split can place an engine's near-identical trajectory in both training and validation data.

This project uses the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark to build a modular workflow for data validation, exploratory analysis, target and feature engineering, model comparison, explainability, experiment tracking, and visual inspection. This paper reports a bounded, reproducible FD001 baseline experiment and distinguishes measured results from components that have only been implemented or unit-tested.

The research question for this experiment is: **How do three conventional regression baselines compare for uncapped cycle-level RUL estimation on FD001 when validation folds are separated by engine?**

## 2. Data

The experiment uses the local `train_FD001.txt` file from the NASA C-MAPSS data package. The loaded training table contains 20,631 observations from 100 engines. Each observation contains three operating settings and 21 sensor measurements; `engine_id` identifies the asset and `cycle` its operating time. The 24 setting/sensor variables form the model input. Engine identifier and cycle are excluded as predictors.

The training RUL target is computed separately for each engine:

`RUL(engine, cycle) = max_cycle(engine) - cycle`

The raw dataset is retained unchanged. The supervised benchmark does not use test-set rows or labels. Although the project contains utilities for loading and labeling test data, those utilities are outside the experiment reported here.

## 3. Methods

### 3.1 Models

The benchmark compares:

1. **Linear regression** with an intercept.
2. **Ridge regression** with the project default regularization strength (`alpha=1.0`).
3. **Random Forest regression** with 300 trees, unrestricted maximum depth, minimum leaf size 2, and random seed 42.

All models use the same raw 24-dimensional input and folds. No hyperparameter search is performed in this baseline experiment. The separate Phase 10 optimization workflow is available for follow-up experiments.

### 3.2 Validation protocol

Three-fold `GroupKFold` cross-validation uses `engine_id` as the grouping variable. For every fold, a model is fitted on the training engines and predicts every observation from held-out engines. Each observation receives one out-of-fold prediction. Metrics in Table 1 are computed on the concatenated out-of-fold predictions, so rows are weighted equally and engines with longer observed trajectories contribute more rows. Reported fold RMSE dispersion is the population standard deviation across the three fold RMSE values; with three folds, this is descriptive and not a confidence interval.

### 3.3 Metrics

We report mean absolute error (MAE), root mean squared error (RMSE), and coefficient of determination (R²). We also report the summed asymmetric NASA-style score, where `d = predicted_RUL - actual_RUL`:

`score(d) = exp(-d / 13) - 1`, if `d < 0`; otherwise `exp(d / 10) - 1`.

The total score is the sum over observations, and lower is better. As a sum, its magnitude depends on the number of evaluated observations; it should only be compared across models evaluated on the same rows. The score is not a per-engine or normalized metric.

## 4. Results

**Table 1. Pooled out-of-fold results on FD001 training data (3 engine-grouped folds).**

| Rank by RMSE | Model | MAE (cycles) | RMSE (cycles) | R² | Summed NASA-style score | Fold RMSE SD |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Random Forest | 30.357 | 42.828 | 0.613 | 192,852,000 | 1.581 |
| 2 | Ridge | 34.473 | 44.974 | 0.574 | 202,107,000 | 2.131 |
| 3 | Linear regression | 34.470 | 44.983 | 0.574 | 204,682,000 | 2.133 |

Random Forest had the lowest pooled RMSE and MAE of the three evaluated models. Ridge and ordinary least squares had nearly identical pooled RMSE. The ordering by the summed asymmetric score is consistent with the RMSE ranking for this run. The fold RMSE standard deviations describe only the three validation partitions and do not establish statistical significance.

The complete unrounded metrics, individual fold metrics, input feature names, and runtime library versions are recorded in [`fd001_raw_sensor_grouped_cv.json`](./fd001_raw_sensor_grouped_cv.json).

## 4.1 Cross-subset robustness protocol

As a follow-up to the FD001 experiment, the identical model definitions, 24 raw input measurements, uncapped training RUL target, and three-fold GroupKFold protocol are applied independently to FD001, FD002, FD003, and FD004. Metrics remain separate for each subset because operating conditions, failure modes, and sample counts differ.

**Table 2. Pooled out-of-fold RMSE by subset, with within-subset rank in parentheses.**

| Subset | Training rows | Engines | Random Forest | Ridge | Linear regression |
|---|---:|---:|---:|---:|---:|
| FD001 | 20,631 | 100 | 42.828 (1) | 44.974 (2) | 44.983 (3) |
| FD002 | 53,759 | 260 | 43.989 (1) | 45.142 (3) | 45.089 (2) |
| FD003 | 24,720 | 100 | 65.460 (3) | 64.438 (1) | 64.576 (2) |
| FD004 | 61,249 | 249 | 57.019 (1) | 60.528 (3) | 60.430 (2) |

Random Forest had the lowest RMSE in three of the four subsets and mean within-subset rank 1.50. Ridge and linear regression each had mean rank 2.25; Ridge ranked first on FD003, while linear regression ranked first on none of the subsets. On FD003, Random Forest had the lowest MAE (43.919 cycles) but the highest RMSE (65.460 cycles), indicating that a subset of larger errors affected its squared-error ranking. These are relative, within-subset comparisons; neither raw RMSE values across datasets nor mean ranks constitute transfer-performance estimates. Full metrics and fold results are in [`cross_subset_grouped_cv.json`](./cross_subset_grouped_cv.json).

## 5. Discussion

The FD001 result supports Random Forest as the strongest of these three specific raw-feature baselines under the stated FD001 grouped-CV protocol. Across the four within-subset runs, Random Forest ranked first by RMSE in three subsets, while Ridge ranked first on FD003. The result does not show that Random Forest is the best model for C-MAPSS in general. The cross-subset experiments evaluate each subset independently; they do not train on one subset and test on another. All runs use default project parameters and a single grouped-CV protocol. They do not assess optimized models, sequence models, uncertainty calibration, or performance by engine.

Grouping by engine reduces the risk of trajectory leakage that would arise from randomly splitting rows. It does not by itself address all sources of evaluation bias. In particular, the reported pooled metrics weight each cycle equally rather than each engine equally, and only three folds are used. Model and feature selection based on these results would require a separate final evaluation, ideally retaining an untouched test set until selection is complete.

The NASA-style score is highly asymmetric and its total is large because it sums nonlinear penalties across thousands of observations. It complements rather than replaces MAE and RMSE; practical comparisons should show the exact formula and whether scores are sums or averages.

## 6. Limitations and responsible interpretation

- The initial result is for FD001 only. Cross-subset experiments do not establish cross-subset transfer because the models are trained and validated within each subset.
- Validation uses three grouped folds; no repeated cross-validation, bootstrap interval, or significance test was performed.
- The target is uncapped training RUL. Capped-label experiments may yield different results.
- Inputs are current-cycle settings and sensor values only; no rolling features, feature selection, or scaling pipeline is used in this reported comparison.
- No trained model artifact or official test-set evaluation is produced by this paper's benchmark.
- C-MAPSS is a simulated benchmark and does not establish operational performance on real aircraft engines.
- Anomaly screening in the dashboard is unsupervised and must not be interpreted as a verified failure diagnosis or as an RUL prediction.

## 7. Conclusion

This paper documents a reproducible, engine-grouped FD001 baseline experiment and an independently evaluated cross-subset extension. On pooled FD001 out-of-fold predictions, the configured Random Forest achieved RMSE 42.828 cycles, lower than Ridge (44.974) and ordinary least squares (44.983). These results are baselines for further controlled study, not a deployment claim. Next evaluations should include repeated grouped validation, per-engine summaries, explicit feature-pipeline ablations, tuned models, cross-subset transfer tests, and a final evaluation on the official held-out test set after model selection.

## Reproducibility

From the project root, with the project dependencies installed:

```powershell
$env:PYTHONPATH='.'
python -m src.analysis.phase13_benchmark --dataset FD001 --folds 3
python -m src.analysis.phase14_cross_subset_benchmark --folds 3
```

The Phase 13 command reads the existing local training file and writes the machine-readable result to `reports/research_paper/fd001_raw_sensor_grouped_cv.json`. The Phase 14 command independently evaluates every available subset and writes `reports/research_paper/cross_subset_grouped_cv.json`. Neither command downloads data or modifies raw files. Run the tests with:

```powershell
python -m pytest -q
```

The broader dashboard is launched with `streamlit run dashboard/app.py`. Optional SHAP, Optuna, and MLflow workflows are documented in the phase-specific notebooks and require their declared dependencies.

## References

1. A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage propagation modeling for aircraft engine run-to-failure simulation,” in *Proceedings of the 2008 International Conference on Prognostics and Health Management*, 2008, doi: [10.1109/PHM.2008.4711414](https://doi.org/10.1109/PHM.2008.4711414).
2. NASA, “C-MAPSS Jet Engine Simulated Data,” NASA Data Catalog. [https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
3. NASA Ames Prognostics Center of Excellence, “Prognostics Data Repository.” [https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).
