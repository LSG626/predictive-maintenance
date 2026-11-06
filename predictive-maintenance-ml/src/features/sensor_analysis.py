from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def select_numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or {'engine_id', 'cycle', 'dataset'}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric if column not in exclude]


def summarize_dataset(df: pd.DataFrame, dataset_name: str | None = None) -> dict[str, Any]:
    """Return a structured EDA summary for a C-MAPSS frame."""
    numeric_cols = select_numeric_columns(df)
    missing = df.isna().sum().to_dict()
    shape = df.shape
    engine_counts = df.groupby('engine_id').size()
    summary = {
        'dataset_name': dataset_name,
        'shape': {'rows': int(shape[0]), 'columns': int(shape[1])},
        'engine_count': int(df['engine_id'].nunique()),
        'cycle_min': int(df['cycle'].min()),
        'cycle_max': int(df['cycle'].max()),
        'duplicate_rows': int(df.duplicated().sum()),
        'missing_values_total': int(sum(missing.values())),
        'missing_by_column': missing,
        'engine_cycle_summary': {
            'min': int(engine_counts.min()),
            'max': int(engine_counts.max()),
            'mean': float(engine_counts.mean()),
            'std': float(engine_counts.std(ddof=0)),
        },
        'descriptive_stats': df[numeric_cols].describe().transpose().to_dict(),
    }
    return summary


def identify_constant_and_near_constant_columns(df: pd.DataFrame, near_constant_limit: int = 2) -> dict[str, list[str]]:
    numeric_cols = select_numeric_columns(df)
    constant_columns = []
    near_constant_columns = []
    for column in numeric_cols:
        unique_count = df[column].nunique(dropna=False)
        if unique_count <= 1:
            constant_columns.append(column)
        elif unique_count <= near_constant_limit:
            near_constant_columns.append(column)
    return {'constant': constant_columns, 'near_constant': near_constant_columns}


def compute_sensor_variance(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = select_numeric_columns(df)
    sensor_variance = df[numeric_cols].var(ddof=0).sort_values(ascending=False).reset_index()
    sensor_variance.columns = ['sensor', 'variance']
    return sensor_variance


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = select_numeric_columns(df)
    return df[numeric_cols].corr()


def compute_pca_projection(df: pd.DataFrame, columns: list[str] | None = None, n_components: int = 2) -> dict[str, Any]:
    numeric_cols = columns or select_numeric_columns(df)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(df[numeric_cols])
    pca = PCA(n_components=min(n_components, matrix.shape[1]))
    components = pca.fit_transform(matrix)
    return {
        'components': components,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'feature_loadings': pca.components_,
        'transformed_columns': numeric_cols,
        'scaler': scaler,
    }


def engine_lifecycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby('engine_id')
        .agg(
            max_cycle=('cycle', 'max'),
            start_cycle=('cycle', 'min'),
            span=('cycle', lambda s: int(s.max() - s.min() + 1)),
        )
        .reset_index()
    )
    return grouped.sort_values('max_cycle').reset_index(drop=True)


def sensor_trend_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = select_numeric_columns(df)
    trend_table = []
    for column in numeric_cols:
        trend_table.append(
            {
                'sensor': column,
                'mean': float(df[column].mean()),
                'std': float(df[column].std(ddof=0)),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'median': float(df[column].median()),
            }
        )
    return pd.DataFrame(trend_table).sort_values('std', ascending=False).reset_index(drop=True)
