from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _resolve_feature_columns(df: pd.DataFrame, feature_columns: list[str] | None = None) -> list[str]:
    if feature_columns is not None:
        return list(feature_columns)
    return [
        column for column in df.columns
        if column not in {'engine_id', 'cycle', 'dataset'}
        and pd.api.types.is_numeric_dtype(df[column])
        and column != 'anomaly_score'
    ]


def detect_isolation_forest(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    feature_columns = _resolve_feature_columns(df, feature_columns)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(df[feature_columns])
    return model


def detect_one_class_svm(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    nu: float = 0.05,
) -> OneClassSVM:
    feature_columns = _resolve_feature_columns(df, feature_columns)
    model = OneClassSVM(nu=nu, gamma='scale')
    model.fit(df[feature_columns])
    return model


def detect_lof(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    contamination: float = 0.05,
    n_neighbors: int = 20,
) -> LocalOutlierFactor:
    feature_columns = _resolve_feature_columns(df, feature_columns)
    model = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors)
    model.fit_predict(df[feature_columns])
    return model


def score_anomalies(
    df: pd.DataFrame,
    model: Any,
    feature_columns: list[str] | None = None,
) -> pd.Series:
    feature_columns = _resolve_feature_columns(df, feature_columns)
    if isinstance(model, IsolationForest):
        score = -model.score_samples(df[feature_columns])
    elif isinstance(model, OneClassSVM):
        score = -model.score_samples(df[feature_columns])
    elif isinstance(model, LocalOutlierFactor):
        score = -model.negative_outlier_factor_
        if score.size == 0:
            score = np.zeros(len(df), dtype=float)
    else:
        raise TypeError(f'Unsupported anomaly model type: {type(model)}')
    return pd.Series(score, index=df.index, name='anomaly_score')


def build_anomaly_summary(df: pd.DataFrame, model: Any, feature_columns: list[str] | None = None) -> pd.DataFrame:
    feature_columns = _resolve_feature_columns(df, feature_columns)
    output = df.copy()
    output['anomaly_score'] = score_anomalies(df, model, feature_columns)
    if isinstance(model, (IsolationForest, OneClassSVM)):
        output['is_anomaly'] = model.predict(output[feature_columns]) == -1
    elif isinstance(model, LocalOutlierFactor):
        pred = model.fit_predict(output[feature_columns])
        output['is_anomaly'] = pred == -1
    return output
