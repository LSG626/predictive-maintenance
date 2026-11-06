# Project Progress

## Phase 0: Initial project scaffold
- Workspace inspected and found empty.
- Created the required project directory structure.
- Added dependency manifest, YAML configuration, and repository hygiene files.
- Added a raw-data note explaining manual download and provenance requirements.

## Phase 1: Data acquisition and validation
- Implemented reusable C-MAPSS dataset loading utilities in `src/data/load_data.py`.
- Implemented dataset integrity and summary validation in `src/data/validate_data.py`.
- Added a project configuration module in `src/utils/config.py`.
- Added targeted tests in `tests/test_phase1.py` covering valid and invalid dataset conditions.

### Validation performed
Command executed:
- `cd ".../predictive-maintenance-ml"; $env:PYTHONPATH='.'; python -m pytest tests/test_phase1.py -q`

Result:
- 2 tests passed

### Important findings
- The full ML stack is intended for Python 3.11; the current local environment is Python 3.14.
- Full dependency installation such as `mlflow`/`pyarrow` is not compatible with Python 3.14 in this environment, so the project requirements are gated to Python < 3.13.
- Phase 1 code is therefore validated using the minimal Python stack required for dataset inspection and integrity checks.

## Phase 2: Advanced exploratory data analysis
- Implemented reusable summary and distribution utilities in `src/features/sensor_analysis.py`.
- Implemented publication-style plotting functions in `src/visualization/plots.py`.
- Added a reusable EDA workflow runner in `src/analysis/phase2_eda.py`.
- Created the notebook skeleton at `notebooks/02_exploratory_data_analysis.ipynb`.
- Added focused tests in `tests/test_phase2.py`.

### Validation performed
Command executed:
- `cd ".../predictive-maintenance-ml"; $env:PYTHONPATH='.'; python -m pytest tests/test_phase2.py -q`

Result:
- 3 tests passed

### Important findings
- The pipeline is ready to run on the actual NASA C-MAPSS raw files once they are placed in `data/raw/`.
- The EDA workflow is modular and can generate missing-value heatmaps, sensor variance summaries, correlation matrices, cycle-distribution charts, and PCA projections without introducing leakage.

## Phase 3: Target engineering and RUL labelling
- Implemented the target engineering utilities in `src/features/target_engineering.py`.
- Added correct training-rule construction using `max_cycle - current_cycle`.
- Added support for capped RUL thresholds and multiple threshold candidates.
- Added correct test-set target reconstruction based on the supplied ground-truth RUL file and the final observed cycle of each engine.
- Added a validation notebook at `notebooks/03_data_preprocessing.ipynb`.
- Added focused tests in `tests/test_phase3.py`.

### Validation performed
Command executed:
- `cd ".../predictive-maintenance-ml"; $env:PYTHONPATH='.'; python -m pytest tests/test_phase3.py -q`

Result:
- 3 tests passed

### Important findings
- The training-label formulation uses the standard C-MAPSS convention, `RUL = max_cycle - current_cycle`.
- The test-set labels are reconstructed using `true_rul_at_final_cycle + final_cycle - current_cycle` to preserve the physical meaning of the supplied ground-truth file.
- The real FD001 dataset yields positive RUL labels for all test rows, confirming the corrected formulation.

## Phase 4: Feature engineering and rolling-window feature generation
- Implemented past-only rolling features, first- and second-order differences, exponential moving averages, and degradation indicators in `src/features/feature_engineering.py`.
- Added interaction features and a reusable feature-construction pipeline for engine-aware prediction tasks.
- Added feature-selection utilities for variance thresholding, correlation filtering, mutual information ranking, and recursive feature elimination.
- Added a validation notebook at `notebooks/04_feature_engineering.ipynb`.
- Added focused tests in `tests/test_phase4.py`.

### Validation performed
Command executed:
- `cd ".../predictive-maintenance-ml"; $env:PYTHONPATH='.'; python -m pytest tests/test_phase4.py -q`

Result:
- 3 tests passed

### Important findings
- Rolling and EWM features remain past-only by applying group-wise transforms and a one-step shift to avoid future leakage.
- The feature-generation pipeline creates a richer representation without altering the final raw dataset files.

## Phase 5: Baseline machine learning models
- Added baseline model factories for linear regression, ridge regression, random forest, SVR, and XGBoost in `src/models/`.
- Added a reusable group-aware cross-validation evaluation suite in `src/models/baseline.py`.
- Added regression metrics in `src/evaluation/metrics.py`.
- Added a baseline-model notebook at `notebooks/05_baseline_models.ipynb`.
- Added focused tests in `tests/test_phase5.py`.

### Validation performed
Command executed:
- `cd ".../predictive-maintenance-ml"; $env:PYTHONPATH='.'; python -m pytest tests/test_phase5.py -q`

Result:
- 2 tests passed

### Important findings
- Engine-aware cross-validation is used to avoid leakage between engine trajectories.
- The baseline model suite is ready to benchmark the feature matrix for FD001 and can be extended to FD002–FD004 with the same modular interface.

## Phase 6: Advanced deep learning models
- Added engine-aware sequence generation utilities in `src/features/sequence_generator.py`.
- Added LSTM, GRU, and 1D CNN model builders in `src/models/lstm.py`, `src/models/gru.py`, and `src/models/cnn.py`.
- Added a deep-learning runner in `src/models/deep_learning.py` to orchestrate sequence creation and model availability checks.
- Added a notebook scaffold at `notebooks/06_deep_learning_models.ipynb`.
- Added focused tests in `tests/test_phase6.py`.

### Validation performed
Command executed:
- `cd ".../predictive-maintenance-ml"; $env:PYTHONPATH='.'; python -m pytest tests/test_phase6.py -q`

Result:
- 2 tests passed

### Important findings
- The deep-learning workflow is structured for TensorFlow-based training, but the current environment does not include the runtime, so the code resolves gracefully with an explicit import error message instead of failing silently.
- The sequence-generation logic is ready for actual model training in a Python 3.11 environment with TensorFlow installed.

## Phase 7: Anomaly detection
- Implemented Isolation Forest, One-Class SVM, and Local Outlier Factor workflows in `src/models/anomaly_detection.py`.
- Added anomaly scoring and summary utilities while restricting model input to numeric sensor/setting features.
- Added focused tests in `tests/test_phase7.py`.

### Validation performed
- `python -m pytest tests/test_phase7.py -q` — 2 tests passed.

## Phase 8: Model explainability
- Added engine-aware holdout permutation importance and optional SHAP global importance in `src/explainability/feature_importance.py`.
- Added reusable feature-importance plotting and a workflow notebook at `notebooks/08_model_explainability.ipynb`.
- Added focused tests in `tests/test_phase8.py`.

### Validation performed
- `python -m pytest tests/test_phase8.py -q` — 3 tests passed.
- Full suite: `python -m pytest -q` — 20 tests passed.
- Confirmed `notebooks/08_model_explainability.ipynb` parses as valid JSON.

## Phase 9: Model evaluation and comparison
- Added the asymmetric NASA C-MAPSS RUL score alongside MAE, RMSE, and R².
- Added pooled out-of-fold model comparison, fold variability reporting, configurable metric ranking, and prediction diagnostic plots in `src/evaluation/comparison.py`.
- Added the evaluation workflow notebook at `notebooks/09_model_evaluation_and_comparison.ipynb` and focused tests in `tests/test_phase9.py`.

### Validation performed
- `python -m pytest -q` — 25 tests passed.
- Confirmed `notebooks/09_model_evaluation_and_comparison.ipynb` parses as valid JSON.

## Phase 10: Hyperparameter optimization
- Added Optuna tuning for Ridge, Random Forest, and RBF-SVR in `src/models/hyperparameter_optimization.py`.
- Every trial uses GroupKFold by engine; supported objectives are MAE, RMSE, R², and NASA C-MAPSS score.
- The result includes trial history, best parameters/value, and the estimator refit on the supplied training data.
- Added the workflow notebook at `notebooks/10_hyperparameter_optimization.ipynb` and focused tests in `tests/test_phase10.py`.

### Validation performed
- `python -m pytest tests/test_phase10.py -q` — 7 tests passed.
- Full suite: `python -m pytest -q` — 32 tests passed.
- Confirmed `notebooks/10_hyperparameter_optimization.ipynb` parses as valid JSON.

## Phase 11: MLOps and experiment tracking
- Added lazy-loaded MLflow logging for parameters, metrics, tags, files/directories, and fitted scikit-learn models in `src/tracking/experiment_tracking.py`.
- Added local tracking configuration and an end-to-end tuning/logging notebook at `notebooks/11_mlflow_experiment_tracking.ipynb`.
- Added focused unit tests in `tests/test_phase11.py`.

### Validation performed
- `python -m pytest tests/test_phase11.py -q` — 3 tests passed.
- Full suite: `python -m pytest -q` — 35 tests passed.
- Confirmed `notebooks/11_mlflow_experiment_tracking.ipynb` parses as valid JSON.
- MLflow itself is not installed in the current Python 3.14 environment; logging calls were verified with a mocked tracking API. The project already declares MLflow for the supported Python 3.11 environment.

## Phase 12: Deployment dashboard
- Added a Streamlit dashboard at `dashboard/app.py` for local dataset selection, engine and sensor exploration, reference RUL plots, data preview/export, and optional Isolation Forest screening.
- Added tested local dataset discovery and loading helpers in `src/dashboard/data.py`; the dashboard does not trigger downloads.
- Clearly labels test RUL as NASA reference ground truth and anomaly flags as unsupervised screening results, not predictions or diagnoses.
- Added focused tests in `tests/test_phase12.py`.

### Validation performed
- `python -m pytest tests/test_phase12.py -q` — 3 tests passed.
- Full suite: `python -m pytest -q` — 38 tests passed.
- `python -m py_compile dashboard\app.py src\dashboard\data.py` — passed.
- Started Streamlit locally and confirmed `/_stcore/health` returned HTTP 200; stopped the test server afterward.

## Phase 13: Research paper and final project documentation
- Added a reproducible FD001 raw-sensor benchmark using three-fold GroupKFold by engine and the existing baseline evaluation suite.
- Ran the benchmark on 20,631 rows from 100 engines; Random Forest ranked first by pooled out-of-fold RMSE (42.828 cycles), followed by Ridge (44.974) and linear regression (44.983).
- Added the manuscript at `reports/research_paper/manuscript.md` and the machine-readable results at `reports/research_paper/fd001_raw_sensor_grouped_cv.json`.
- Updated the root README with environment setup, benchmark/dashboard commands, scope, and research-report links.

### Validation performed
- `python -m pytest tests/test_phase13.py -q` — 2 tests passed.
- Full suite: `python -m pytest -q` — 40 tests passed.
- Reproduced the FD001 three-fold grouped benchmark with `python -m src.analysis.phase13_benchmark --dataset FD001 --folds 3`; JSON results were regenerated successfully.
- `python -m py_compile src\analysis\phase13_benchmark.py` — passed.

## Phase 14: Cross-subset robustness
- Added a reusable runner applying the same three raw-sensor baselines and GroupKFold-by-engine protocol independently across FD001–FD004.
- Reports subset-specific pooled out-of-fold metrics plus mean within-subset ranks; performance metrics are never pooled across the differing subsets.
- Ran the comparison on all four local subsets. Random Forest ranked first by RMSE on FD001, FD002, and FD004; Ridge ranked first on FD003. Mean within-subset ranks were Random Forest 1.50, Ridge 2.25, and linear regression 2.25.
- Persisted full results in `reports/research_paper/cross_subset_grouped_cv.json` and updated the research manuscript and README with measured results and interpretation limits.
- Added focused input-validation and rank-summary tests in `tests/test_phase14.py`.

### Validation performed
- `python -m pytest tests/test_phase14.py -q` — 3 tests passed.
- Full suite: `python -m pytest -q` — 43 tests passed.
- Ran `python -m src.analysis.phase14_cross_subset_benchmark --folds 3` successfully on all local C-MAPSS training subsets.
- `python -m py_compile src\analysis\phase14_cross_subset_benchmark.py` — passed.
