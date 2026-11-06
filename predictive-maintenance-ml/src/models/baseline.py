from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import regression_metrics
from src.models.random_forest import build_random_forest_model
from src.models.svr import build_svr_model
from src.models.xgboost_model import build_xgboost_model


@dataclass
class BaselineModelResult:
    model_name: str
    fold_metrics: list[dict[str, float]]
    mean_mae: float
    mean_rmse: float
    mean_r2: float
    predictions: list[float]
    true_values: list[float]


def build_linear_regression_model(**kwargs: Any) -> LinearRegression:
    params = {'fit_intercept': True}
    params.update(kwargs)
    return LinearRegression(**params)


def build_ridge_model(**kwargs: Any) -> Ridge:
    params = {'alpha': 1.0, 'random_state': 42}
    params.update(kwargs)
    return Ridge(**params)


def select_feature_columns(df: pd.DataFrame, target_column: str = 'rul') -> list[str]:
    excluded = {'engine_id', 'cycle', 'dataset', target_column}
    return [
        column for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def compute_group_aware_cv_metrics(
    df: pd.DataFrame,
    target_column: str,
    model_factory: Callable[[], Any],
    feature_columns: Sequence[str] | None = None,
    n_splits: int = 3,
    group_column: str = 'engine_id',
) -> BaselineModelResult:
    """Evaluate a model with GroupKFold to avoid leakage across engine trajectories."""
    if feature_columns is None:
        feature_columns = select_feature_columns(df, target_column=target_column)
    if not feature_columns:
        raise ValueError('No numeric feature columns were found for model evaluation.')

    X = df[list(feature_columns)].copy()
    y = df[target_column].astype(float).to_numpy()
    groups = df[group_column].to_numpy()
    n_splits = min(n_splits, int(df[group_column].nunique()))
    if n_splits < 2:
        raise ValueError('At least two unique engines are required for cross-validation.')

    cv = GroupKFold(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []
    predictions: list[float] = []
    true_values: list[float] = []

    for train_idx, test_idx in cv.split(X, y, groups):
        model = model_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        fold_predictions = model.predict(X.iloc[test_idx])
        fold_true = y[test_idx]
        metrics = regression_metrics(fold_true, fold_predictions)
        fold_metrics.append(metrics)
        predictions.extend(fold_predictions.tolist())
        true_values.extend(fold_true.tolist())

    overall_metrics = regression_metrics(np.asarray(true_values), np.asarray(predictions))
    return BaselineModelResult(
        model_name=model_factory.__name__ if hasattr(model_factory, '__name__') else 'model',
        fold_metrics=fold_metrics,
        mean_mae=float(np.mean([m['mae'] for m in fold_metrics])),
        mean_rmse=float(np.mean([m['rmse'] for m in fold_metrics])),
        mean_r2=float(np.mean([m['r2'] for m in fold_metrics])),
        predictions=predictions,
        true_values=true_values,
    )


def run_baseline_model_suite(
    df: pd.DataFrame,
    target_column: str = 'rul',
    feature_columns: Sequence[str] | None = None,
    n_splits: int = 3,
    group_column: str = 'engine_id',
    include_models: Sequence[str] | None = None,
) -> dict[str, BaselineModelResult]:
    """Run the standard baseline model suite using engine-aware cross-validation."""
    if include_models is None:
        include_models = ['linear_regression', 'ridge', 'random_forest', 'svr']
        try:
            build_xgboost_model()
        except ImportError:
            pass
        else:
            include_models = list(include_models) + ['xgboost']

    model_factories = {
        'linear_regression': build_linear_regression_model,
        'ridge': build_ridge_model,
        'random_forest': build_random_forest_model,
        'svr': build_svr_model,
        'xgboost': build_xgboost_model,
    }

    results: dict[str, BaselineModelResult] = {}
    for model_name in include_models:
        if model_name not in model_factories:
            continue
        model_factory = model_factories[model_name]
        result = compute_group_aware_cv_metrics(
            df=df,
            target_column=target_column,
            model_factory=model_factory,
            feature_columns=feature_columns,
            n_splits=n_splits,
            group_column=group_column,
        )
        result.model_name = model_name
        results[model_name] = result
    return results
