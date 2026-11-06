from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, VarianceThreshold, mutual_info_regression
from sklearn.linear_model import LinearRegression


def _base_feature_columns(df: pd.DataFrame, target_column: str | None = None) -> list[str]:
    exclude = {'engine_id', 'cycle', 'dataset'}
    if target_column is not None:
        exclude.add(target_column)
    return [column for column in df.columns if column not in exclude and pd.api.types.is_numeric_dtype(df[column])]


def add_rolling_features(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    windows: tuple[int, ...] = (3, 5, 10),
    target_column: str | None = None,
) -> pd.DataFrame:
    """Create past-only rolling statistics using engine-aware windows.

    Rolling features are computed with a groupby transform on `engine_id` and shifted by one step to
    avoid using future observations within the same engine trajectory.
    """
    out = df.copy()
    feature_columns = list(columns) if columns is not None else _base_feature_columns(out, target_column)
    for column in feature_columns:
        for window in windows:
            out[f'{column}_roll_mean_{window}'] = (
                out.groupby('engine_id')[column].transform(
                    lambda s: s.rolling(window, min_periods=1).mean().shift(1).fillna(0)
                )
            )
            out[f'{column}_roll_std_{window}'] = (
                out.groupby('engine_id')[column].transform(
                    lambda s: s.rolling(window, min_periods=1).std().shift(1).fillna(0)
                )
            )
            out[f'{column}_roll_min_{window}'] = (
                out.groupby('engine_id')[column].transform(
                    lambda s: s.rolling(window, min_periods=1).min().shift(1).fillna(0)
                )
            )
            out[f'{column}_roll_max_{window}'] = (
                out.groupby('engine_id')[column].transform(
                    lambda s: s.rolling(window, min_periods=1).max().shift(1).fillna(0)
                )
            )

        out[f'{column}_diff_1'] = out.groupby('engine_id')[column].transform(lambda s: s.diff().fillna(0))
        out[f'{column}_diff_2'] = out.groupby('engine_id')[column].transform(lambda s: s.diff().diff().fillna(0))
        out[f'{column}_ewm'] = out.groupby('engine_id')[column].transform(
            lambda s: s.ewm(alpha=0.2, min_periods=1, adjust=False).mean().shift(1).fillna(0)
        )
        out[f'{column}_trend'] = out[f'{column}_diff_1'] / out.groupby('engine_id')[column].transform(
            lambda s: s.rolling(5, min_periods=1).mean().shift(1).replace(0, np.nan)
        )
        out[f'{column}_trend'] = out[f'{column}_trend'].replace([np.inf, -np.inf], 0).fillna(0)

    return out


def add_interaction_features(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Create simple ratio-based sensor interaction features using a limited subset of sensors."""
    out = df.copy()
    feature_columns = list(columns) if columns is not None else _base_feature_columns(out)
    primary_columns = [column for column in feature_columns if column.startswith('sensor_')][:6]
    for left_index, left in enumerate(primary_columns):
        for right in primary_columns[left_index + 1 :]:
            safe_right = out[right].replace(0, np.nan)
            out[f'{left}_div_{right}'] = out[left] / safe_right
            out[f'{left}_div_{right}'] = out[f'{left}_div_{right}'].replace([np.inf, -np.inf], 0).fillna(0)
    return out


def add_degradation_indicators(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Add simple degradation features that summarise sensor drift relative to the engine baseline."""
    out = df.copy()
    feature_columns = list(columns) if columns is not None else _base_feature_columns(out)
    new_columns: dict[str, pd.Series] = {}
    for column in feature_columns:
        if column.startswith('sensor_'):
            baseline = out.groupby('engine_id')[column].transform('mean')
            new_columns[f'{column}_deviation_from_mean'] = out[column] - baseline
            new_columns[f'{column}_normalized_cycle'] = out[column] / out['cycle']
    return pd.concat([out, pd.DataFrame(new_columns)], axis=1)


def build_feature_matrix(
    df: pd.DataFrame,
    target_column: str = 'rul',
    windows: tuple[int, ...] = (3, 5, 10),
) -> pd.DataFrame:
    """Engineer a feature matrix suitable for regression with engine-aware, past-only features."""
    out = df.copy()
    if target_column in out.columns:
        out = out.copy()
    else:
        raise ValueError(f'{target_column} must be present in the input DataFrame.')

    out = add_rolling_features(out, target_column=target_column, windows=windows)
    out = add_interaction_features(out)
    out = add_degradation_indicators(out)
    return out


def select_features_by_variance_threshold(df: pd.DataFrame, target_column: str = 'rul', threshold: float = 0.0) -> list[str]:
    """Apply a variance threshold to retain informative features only."""
    feature_columns = _base_feature_columns(df, target_column)
    if not feature_columns:
        return []

    X = df[feature_columns].copy()
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    return [column for column, keep in zip(feature_columns, selector.get_support()) if keep]


def select_features_by_correlation(df: pd.DataFrame, target_column: str = 'rul', min_abs_corr: float = 0.05) -> list[str]:
    """Retain features whose absolute correlation with the target exceeds a threshold."""
    feature_columns = _base_feature_columns(df, target_column)
    if not feature_columns:
        return []

    corr = df[feature_columns + [target_column]].corr(numeric_only=True)[target_column]
    return [column for column in feature_columns if abs(corr.get(column, 0.0)) >= min_abs_corr]


def select_features_by_mutual_information(df: pd.DataFrame, target_column: str = 'rul', top_k: int = 10, random_state: int = 42) -> list[str]:
    """Rank features by mutual information and keep the top-k predictors."""
    feature_columns = _base_feature_columns(df, target_column)
    if not feature_columns:
        return []

    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)
    scores = mutual_info_regression(X, y, random_state=random_state)
    ranked = sorted(zip(feature_columns, scores), key=lambda pair: pair[1], reverse=True)
    return [column for column, _ in ranked[:top_k]]


def select_features_by_rfe(df: pd.DataFrame, target_column: str = 'rul', n_features: int = 10) -> list[str]:
    """Use recursive feature elimination to keep a compact feature subset."""
    feature_columns = _base_feature_columns(df, target_column)
    if not feature_columns:
        return []

    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)
    model = LinearRegression()
    selector = RFE(estimator=model, n_features_to_select=min(n_features, X.shape[1]))
    selector.fit(X, y)
    return [feature_columns[idx] for idx in np.where(selector.support_)[0]]
