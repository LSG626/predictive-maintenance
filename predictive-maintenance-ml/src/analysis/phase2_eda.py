from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_data import collect_fd_dataset
from src.data.validate_data import validate_dataset
from src.features.sensor_analysis import (
    compute_correlation_matrix,
    compute_pca_projection,
    compute_sensor_variance,
    engine_lifecycle_summary,
    identify_constant_and_near_constant_columns,
    select_numeric_columns,
    sensor_trend_summary,
    summarize_dataset,
)
from src.visualization.plots import (
    plot_correlation_heatmap,
    plot_cycle_distribution,
    plot_engine_lifecycle,
    plot_missing_value_heatmap,
    plot_pca_projection,
    plot_sensor_distribution,
    plot_sensor_variance,
    plot_sensor_time_series,
)


def run_phase2_eda(dataset_name: str = 'FD001', raw_dir: str | Path = 'data/raw', output_dir: str | Path = 'reports/figures/phase2') -> dict[str, Any]:
    """Run the phase 2 EDA workflow for one FD subset.

    The function expects the raw C-MAPSS files to exist in the raw data directory.
    """
    raw_data_dir = Path(raw_dir)
    train_path = raw_data_dir / f'train_{dataset_name}.txt'
    test_path = raw_data_dir / f'test_{dataset_name}.txt'
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f'Missing C-MAPSS raw files for {dataset_name}. '
            f'Put train_{dataset_name}.txt and test_{dataset_name}.txt into {raw_data_dir}. '
            'The workflow requires the original raw data files to remain unchanged.'
        )

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, engine='python')
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, engine='python')

    columns = ['engine_id', 'cycle'] + [f'setting_{idx}' for idx in range(1, 4)] + [f'sensor_{idx}' for idx in range(1, 22)]
    train_df.columns = columns
    test_df.columns = columns
    train_df['dataset'] = dataset_name
    test_df['dataset'] = dataset_name

    validate_dataset(train_df)
    validate_dataset(test_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'train_summary': summarize_dataset(train_df, dataset_name=f'train_{dataset_name}'),
        'test_summary': summarize_dataset(test_df, dataset_name=f'test_{dataset_name}'),
        'constant_columns': identify_constant_and_near_constant_columns(train_df),
        'variance_top_sensors': compute_sensor_variance(train_df).head(10).to_dict(orient='records'),
        'correlation_matrix': compute_correlation_matrix(train_df).to_dict(),
        'engine_lifecycle_summary': engine_lifecycle_summary(train_df).to_dict(orient='records'),
        'sensor_trend_summary': sensor_trend_summary(train_df).to_dict(orient='records'),
        'pca_projection': compute_pca_projection(train_df),
    }

    plot_missing_value_heatmap(train_df, output_dir / 'missing_values_train.png')
    plot_sensor_distribution(train_df, output_dir / 'sensor_distribution_train.png', top_n=6)
    plot_sensor_variance(train_df, output_dir / 'sensor_variance_train.png', top_n=12)
    plot_correlation_heatmap(train_df, output_dir / 'sensor_correlation_matrix.png')
    plot_engine_lifecycle(train_df, output_dir / 'sample_engine_lifecycles.png', sample_size=12)
    plot_cycle_distribution(train_df, output_dir / 'cycle_count_distribution.png')
    plot_pca_projection(train_df, output_dir / 'pca_projection.png')
    plot_sensor_time_series(train_df, output_dir / 'sensor_2_over_cycles.png', sensor_name='sensor_2')

    return report
