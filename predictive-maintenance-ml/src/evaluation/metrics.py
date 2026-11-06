from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def nasa_rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the asymmetric NASA C-MAPSS score; lower values are better."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0:
        raise ValueError('RUL score requires at least one observation.')
    if actual.shape != predicted.shape:
        raise ValueError('y_true and y_pred must have the same number of observations.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('y_true and y_pred must contain only finite values.')

    error = predicted - actual
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    # Keep unusually large errors finite while preserving the score's scale.
    return float(np.expm1(np.minimum(exponent, 700.0)).sum())


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'nasa_score': nasa_rul_score(y_true, y_pred),
    }
