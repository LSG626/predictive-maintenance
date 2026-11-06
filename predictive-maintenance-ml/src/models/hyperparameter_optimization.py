from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR

from src.evaluation.metrics import regression_metrics
from src.models.baseline import select_feature_columns

_SUPPORTED_METRICS = {'mae', 'rmse', 'r2', 'nasa_score'}


@dataclass
class HyperparameterOptimizationResult:
    model_name: str
    metric: str
    best_value: float
    best_params: dict[str, Any]
    best_model: Any
    trials: pd.DataFrame


def _suggest_parameters(trial: Any, model_name: str) -> dict[str, Any]:
    if model_name == 'ridge':
        return {'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True)}
    if model_name == 'random_forest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 20, 30]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        }
    if model_name == 'svr':
        return {
            'C': trial.suggest_float('C', 1e-1, 1e3, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-2, 1.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        }
    raise ValueError(f'Unsupported model_name {model_name!r}; choose from ridge, random_forest, or svr.')


def _build_model(model_name: str, params: dict[str, Any], random_state: int) -> Any:
    if model_name == 'ridge':
        return Ridge(**params)
    if model_name == 'random_forest':
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    if model_name == 'svr':
        return SVR(kernel='rbf', **params)
    raise ValueError(f'Unsupported model_name {model_name!r}; choose from ridge, random_forest, or svr.')


def optimize_hyperparameters(
    df: pd.DataFrame,
    model_name: str = 'random_forest',
    target_column: str = 'rul',
    feature_columns: Sequence[str] | None = None,
    group_column: str = 'engine_id',
    metric: str = 'rmse',
    n_splits: int = 3,
    n_trials: int = 30,
    random_state: int = 42,
    timeout: float | None = None,
) -> HyperparameterOptimizationResult:
    """Optimize a regressor against engine-grouped CV without engine leakage."""
    if model_name not in {'ridge', 'random_forest', 'svr'}:
        raise ValueError('model_name must be one of ridge, random_forest, or svr.')
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f'metric must be one of {sorted(_SUPPORTED_METRICS)}.')
    if n_trials < 1:
        raise ValueError('n_trials must be at least 1.')
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2.')
    required = {target_column, group_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'DataFrame is missing required columns: {sorted(missing)}.')
    if feature_columns is None:
        feature_columns = select_feature_columns(df, target_column=target_column)
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError('At least one feature column is required.')
    missing_features = set(feature_columns).difference(df.columns)
    if missing_features:
        raise ValueError(f'DataFrame is missing feature columns: {sorted(missing_features)}.')

    X = df[feature_columns].copy()
    y = df[target_column].astype(float).to_numpy()
    groups = df[group_column].to_numpy()
    if X.empty or X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Features must contain at least one row and only finite, non-missing values.')
    if not np.isfinite(y).all():
        raise ValueError('Target values must be finite.')
    n_groups = pd.Series(groups).nunique()
    actual_splits = min(n_splits, int(n_groups))
    if actual_splits < 2:
        raise ValueError('At least two unique groups are required for cross-validation.')

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            'Hyperparameter optimization requires Optuna. Install the project dependencies '
            'in a supported Python environment (for example, pip install optuna).'
        ) from exc

    cv = GroupKFold(n_splits=actual_splits)
    direction = 'maximize' if metric == 'r2' else 'minimize'
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial: Any) -> float:
        params = _suggest_parameters(trial, model_name)
        fold_scores: list[float] = []
        for train_idx, valid_idx in cv.split(X, y, groups):
            model = _build_model(model_name, params, random_state)
            model.fit(X.iloc[train_idx], y[train_idx])
            predictions = model.predict(X.iloc[valid_idx])
            fold_scores.append(regression_metrics(y[valid_idx], predictions)[metric])
        return float(np.mean(fold_scores))

    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    if not study.trials or study.best_trial is None:
        raise RuntimeError('Optuna completed without a successful optimization trial.')

    best_params = dict(study.best_params)
    best_model = _build_model(model_name, best_params, random_state)
    best_model.fit(X, y)

    trial_rows = []
    for trial in study.trials:
        row: dict[str, Any] = {
            'trial': trial.number,
            'value': trial.value,
            'state': trial.state.name,
        }
        row.update(trial.params)
        trial_rows.append(row)
    trials = pd.DataFrame(trial_rows)
    return HyperparameterOptimizationResult(
        model_name=model_name,
        metric=metric,
        best_value=float(study.best_value),
        best_params=best_params,
        best_model=best_model,
        trials=trials,
    )
