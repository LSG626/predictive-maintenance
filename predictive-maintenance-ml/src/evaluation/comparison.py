from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics

if TYPE_CHECKING:
    from src.models.baseline import BaselineModelResult

_METRIC_DIRECTIONS = {'mae': 'min', 'rmse': 'min', 'r2': 'max', 'nasa_score': 'min'}


def compare_model_results(
    results: Mapping[str, BaselineModelResult],
    rank_by: str = 'rmse',
) -> pd.DataFrame:
    """Compare pooled out-of-fold predictions and fold variability across models."""
    if not results:
        raise ValueError('At least one model result is required.')
    if rank_by not in _METRIC_DIRECTIONS:
        raise ValueError(f'rank_by must be one of {sorted(_METRIC_DIRECTIONS)}.')

    rows: list[dict[str, float | int | str]] = []
    for model_name, result in results.items():
        actual = np.asarray(result.true_values, dtype=float)
        predicted = np.asarray(result.predictions, dtype=float)
        if actual.size == 0 or actual.shape != predicted.shape:
            raise ValueError(
                f'Model {model_name!r} must have non-empty, equally sized true_values and predictions.'
            )
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError(f'Model {model_name!r} predictions and true values must be finite.')

        metrics = regression_metrics(actual, predicted)
        fold_metric_values = {
            metric: np.asarray([fold[metric] for fold in result.fold_metrics if metric in fold], dtype=float)
            for metric in ('mae', 'rmse', 'r2')
        }
        rows.append({
            'model': model_name,
            **metrics,
            'mae_std': float(fold_metric_values['mae'].std(ddof=0)) if fold_metric_values['mae'].size else 0.0,
            'rmse_std': float(fold_metric_values['rmse'].std(ddof=0)) if fold_metric_values['rmse'].size else 0.0,
            'r2_std': float(fold_metric_values['r2'].std(ddof=0)) if fold_metric_values['r2'].size else 0.0,
        })

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        rank_by,
        ascending=_METRIC_DIRECTIONS[rank_by] == 'min',
        ignore_index=True,
    )
    comparison.insert(0, 'rank', np.arange(1, len(comparison) + 1))
    return comparison


def plot_prediction_diagnostics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    output_path: str | Path,
    title: str = 'Regression prediction diagnostics',
) -> Path:
    """Save actual-versus-predicted and residual-versus-predicted plots."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError('y_true and y_pred must be non-empty and have the same length.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('y_true and y_pred must contain only finite values.')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    residuals = predicted - actual
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(actual, predicted, alpha=0.65)
    axes[0].plot([lower, upper], [lower, upper], linestyle='--', color='black')
    axes[0].set_title('Actual vs predicted')
    axes[0].set_xlabel('Actual RUL')
    axes[0].set_ylabel('Predicted RUL')

    axes[1].scatter(predicted, residuals, alpha=0.65)
    axes[1].axhline(0, linestyle='--', color='black')
    axes[1].set_title('Residuals vs predicted')
    axes[1].set_xlabel('Predicted RUL')
    axes[1].set_ylabel('Predicted - actual')
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path
