# Predictive Maintenance Through Multivariate Time-Series Machine Learning

This project implements a research-oriented predictive-maintenance workflow for the NASA C-MAPSS turbofan engine degradation dataset. It includes data validation and exploratory analysis, RUL and feature engineering, regression baselines, deep-learning model scaffolds, anomaly screening, explainability, grouped evaluation and tuning, MLflow integration, and a Streamlit dashboard. The FD001 baseline findings and study limitations are documented in [`reports/research_paper/manuscript.md`](reports/research_paper/manuscript.md).

## Recommended environment
- Python 3.11 (primary supported runtime for the full stack)
- Jupyter Notebook
- A standard laptop or desktop with at least 8 GB RAM

## Compatibility note
The full research pipeline depends on packages such as `mlflow` and `pyarrow`, which do not install consistently on Python 3.14 in the current environment. For a fully reproducible deployment, use Python 3.11 for the complete project stack. The current local validation environment is Python 3.14, so Phase 1 and Phase 2 validation were performed using the minimal package set required for dataset validation and EDA utilities.

## Project structure
See the repository layout under `predictive-maintenance-ml/` for data, source, notebook, dashboard, and reporting directories.

## Current project maturity
- Phase 0: complete
- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete
- Phase 5: complete
- Phase 6: complete
- Phase 7: complete
- Phase 8: complete
- Phase 9: complete
- Phase 10: complete
- Phase 11: complete
- Phase 12: complete
- Phase 13: complete
- Phase 14: complete

## Getting started
1. Install Python 3.11 and create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install the project dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Place the original C-MAPSS files in `data/raw/CMAPSSData/` (or use an existing local copy). See [`data/raw/README.md`](data/raw/README.md).
4. From the project root, run tests and the local FD001 baseline:
   ```powershell
   $env:PYTHONPATH='.'
   python -m pytest -q
   python -m src.analysis.phase13_benchmark --dataset FD001 --folds 3
   python -m src.analysis.phase14_cross_subset_benchmark --folds 3
   ```
5. Launch the local dashboard:
   ```powershell
   streamlit run dashboard/app.py
   ```

## Phase 2 capabilities
The EDA workflow includes:
- descriptive statistics and dataset overview
- missing-value heatmaps
- sensor variance analysis
- correlation matrices
- lifecycle summaries across engines
- cycle-count distributions
- PCA projection visualisations
- sample sensor trajectory plots

## Phase 8 capabilities
- Engine-aware holdout permutation importance for RUL regression models.
- Optional SHAP-based global feature importance for fitted models.
- Reusable importance plots saved under `reports/`.
- Notebook workflow at `notebooks/08_model_explainability.ipynb`.

## Phase 9 capabilities
- Pooled out-of-fold regression comparison with MAE, RMSE, R², and the asymmetric NASA C-MAPSS score.
- Fold-to-fold metric variability and configurable model ranking.
- Actual-vs-predicted and residual diagnostic plots.
- Notebook workflow at `notebooks/09_model_evaluation_and_comparison.ipynb`.

## Phase 10 capabilities
- Optuna tuning for Ridge, Random Forest, and RBF-SVR regressors.
- Engine-grouped cross-validation for every trial to prevent trajectory leakage.
- Configurable objective metric (MAE, RMSE, R², or NASA score), reproducible TPE sampling, trial export, and refit of the selected estimator.
- Notebook workflow at `notebooks/10_hyperparameter_optimization.ipynb`.

## Phase 11 capabilities
- MLflow tracking for model parameters, metrics, tags, artifacts, and fitted scikit-learn models.
- Local file-based tracking configured in `configs/config.yaml`; no tracking server is required.
- Notebook workflow at `notebooks/11_mlflow_experiment_tracking.ipynb`.
- Launch the local experiment browser with `mlflow ui --backend-store-uri ./mlruns`.

## Phase 12 capabilities
- Streamlit dashboard for locally available C-MAPSS subsets, with engine/sensor trajectories, reference RUL, dataset preview, and CSV export.
- Optional Isolation Forest screening trained on the training split and applied to the selected split.
- Run from the project root with `streamlit run dashboard/app.py`.
- Anomaly flags are exploratory screening signals, not confirmed failures or RUL model predictions.
- A **Research & Results** page is available in the Streamlit navigation, with interactive benchmark comparisons, fold-level results, methodology and limitations, and the full manuscript. It reads the checked-in JSON result files under `reports/research_paper/`.

## Research report
- [`reports/research_paper/manuscript.md`](reports/research_paper/manuscript.md) describes the FD001 baseline and cross-subset grouped-CV experiments, measured results, methods, limitations, and references.
- [`reports/research_paper/fd001_raw_sensor_grouped_cv.json`](reports/research_paper/fd001_raw_sensor_grouped_cv.json) stores the full result table and environment versions. Recreate it with the Phase 13 benchmark command above.
- Phase 14 evaluates the same baselines independently across FD001–FD004; its metrics are not pooled across subsets. Recreate the run with `python -m src.analysis.phase14_cross_subset_benchmark --folds 3`; results are written to `reports/research_paper/cross_subset_grouped_cv.json`.

## Notes
- The project keeps original raw data files intact.
- The reported benchmark is a training-set cross-validation comparison, not an official test-set score or evidence of real-world aircraft performance.
- TensorFlow and MLflow workflows require the supported Python 3.11 environment; optional integrations should be treated as unavailable until their runtime dependencies are installed and validated.
- The workflow is modular and designed for incremental research phases.
