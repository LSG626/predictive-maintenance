from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = (
    ['engine_id', 'cycle']
    + [f'setting_{idx}' for idx in range(1, 4)]
    + [f'sensor_{idx}' for idx in range(1, 22)]
)


def validate_dataset_columns(df: pd.DataFrame) -> list[str]:
    """Check for missing required columns and return the missing ones."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    return missing


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Produce a structured validation report for a C-MAPSS frame."""
    if df.empty:
        raise ValueError('Dataset is empty.')

    missing_columns = validate_dataset_columns(df)
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    df_clean = df.copy()
    df_clean['engine_id'] = pd.to_numeric(df_clean['engine_id'], errors='raise')
    df_clean['cycle'] = pd.to_numeric(df_clean['cycle'], errors='raise')

    missing_values = df_clean.isna().sum().to_dict()
    duplicate_rows = int(df_clean.duplicated().sum())
    engine_counts = df_clean.groupby('engine_id').size().sort_values()

    summary = {
        'n_rows': int(df_clean.shape[0]),
        'n_columns': int(df_clean.shape[1]),
        'n_engines': int(df_clean['engine_id'].nunique()),
        'min_cycle': int(df_clean['cycle'].min()),
        'max_cycle': int(df_clean['cycle'].max()),
        'duplicate_rows': duplicate_rows,
        'missing_values_total': int(sum(missing_values.values())),
        'missing_values_by_column': missing_values,
        'engine_counts': engine_counts.to_dict(),
        'avg_rows_per_engine': float(df_clean.groupby('engine_id').size().mean()),
        'std_rows_per_engine': float(df_clean.groupby('engine_id').size().std(ddof=0)),
    }
    return summary


def dataset_integrity_report(df: pd.DataFrame) -> dict[str, Any]:
    """Return a complete integrity report for the dataset."""
    summary = validate_dataset(df)
    summary['constant_columns'] = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]
    summary['near_constant_columns'] = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 2
    ]
    return summary


def describe_engine_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the lifecycle length for each engine."""
    grouped = df.groupby('engine_id')['cycle'].max().reset_index().rename(columns={'cycle': 'max_cycle'})
    return grouped.sort_values('max_cycle').reset_index(drop=True)
