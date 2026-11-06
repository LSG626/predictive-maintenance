from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features.sensor_analysis import compute_correlation_matrix, compute_sensor_variance, select_numeric_columns


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_missing_value_heatmap(df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.isna().astype(int), cmap='viridis', cbar_kws={'label': 'Missing value flag'}, ax=ax)
    ax.set_title('Missing values heatmap')
    path = Path(output_path)
    _savefig(fig, path)
    return path


def plot_sensor_distribution(df: pd.DataFrame, output_path: str | Path, top_n: int = 6) -> Path:
    numeric_cols = select_numeric_columns(df)[:top_n]
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, column in zip(axes, numeric_cols):
        sns.histplot(df[column], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
    path = Path(output_path)
    _savefig(fig, path)
    return path


def plot_sensor_variance(df: pd.DataFrame, output_path: str | Path, top_n: int = 12) -> Path:
    variance_df = compute_sensor_variance(df).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=variance_df, x='variance', y='sensor', hue='sensor', dodge=False, palette='viridis', legend=False, ax=ax)
    ax.set_title(f'Top {top_n} sensors by variance')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Sensor')
    path = Path(output_path)
    _savefig(fig, path)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str | Path) -> Path:
    corr = compute_correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, annot=False, ax=ax)
    ax.set_title('Sensor correlation matrix')
    path = Path(output_path)
    _savefig(fig, path)
    return path


def plot_engine_lifecycle(df: pd.DataFrame, output_path: str | Path, sample_size: int = 12) -> Path:
    sample_engines = df['engine_id'].drop_duplicates().head(sample_size).tolist()
    sample_df = df[df['engine_id'].isin(sample_engines)].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for engine_id, engine_data in sample_df.groupby('engine_id'):
        ax.plot(engine_data['cycle'], engine_data['sensor_2'], label=f'Engine {engine_id}', alpha=0.8)
    ax.set_title('Sample engine degradation trajectories')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Sensor 2')
    ax.legend(loc='upper right', fontsize=8)
    path = Path(output_path)
    _savefig(fig, path)
    return path


def plot_cycle_distribution(df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df.groupby('engine_id').size(), bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of cycle counts per engine')
    ax.set_xlabel('Number of observed cycles')
    path = Path(output_path)
    _savefig(fig, path)
    return path


def plot_pca_projection(df: pd.DataFrame, output_path: str | Path, columns: list[str] | None = None) -> Path:
    from src.features.sensor_analysis import compute_pca_projection

    pca_result = compute_pca_projection(df, columns=columns)
    projected = pca_result['components']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.7)
    ax.set_title('PCA projection of sensor space')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    path = Path(output_path)
    _savefig(fig, path)
    return path


def plot_sensor_time_series(df: pd.DataFrame, output_path: str | Path, sensor_name: str = 'sensor_2') -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    for engine_id, engine_data in df.groupby('engine_id'):
        ax.plot(engine_data['cycle'], engine_data[sensor_name], alpha=0.4, linewidth=1)
    ax.set_title(f'{sensor_name} over operating cycles across engines')
    ax.set_xlabel('Cycle')
    ax.set_ylabel(sensor_name)
    path = Path(output_path)
    _savefig(fig, path)
    return path
