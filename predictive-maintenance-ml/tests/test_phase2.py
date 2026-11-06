from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.sensor_analysis import (
    compute_correlation_matrix,
    compute_pca_projection,
    compute_sensor_variance,
    engine_lifecycle_summary,
    identify_constant_and_near_constant_columns,
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


def _make_sample_frame() -> pd.DataFrame:
    rows = []
    for engine in [1, 2, 3]:
        for cycle in range(1, 6):
            row = {'engine_id': engine, 'cycle': cycle, 'setting_1': 50.0 + cycle, 'setting_2': 40.0, 'setting_3': 30.0}
            for sensor in range(1, 22):
                row[f'sensor_{sensor}'] = float(sensor + engine * 3 + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


def test_summarize_dataset_and_variance(tmp_path: Path) -> None:
    df = _make_sample_frame()
    summary = summarize_dataset(df)
    variance = compute_sensor_variance(df)
    assert summary['engine_count'] == 3
    assert summary['shape']['rows'] == 15
    assert not variance.empty
    assert variance['variance'].gt(0).sum() > 0


def test_phase2_eda_visuals(tmp_path: Path) -> None:
    df = _make_sample_frame()
    plot_missing_value_heatmap(df, tmp_path / 'missing.png')
    plot_sensor_distribution(df, tmp_path / 'sensor_distribution.png', top_n=3)
    plot_sensor_variance(df, tmp_path / 'sensor_variance.png', top_n=5)
    plot_correlation_heatmap(df, tmp_path / 'corr.png')
    plot_engine_lifecycle(df, tmp_path / 'engine_lifecycle.png', sample_size=2)
    plot_cycle_distribution(df, tmp_path / 'cycle_distribution.png')
    plot_pca_projection(df, tmp_path / 'pca.png')
    plot_sensor_time_series(df, tmp_path / 'time_series.png', sensor_name='sensor_2')

    output_files = [
        tmp_path / 'missing.png',
        tmp_path / 'sensor_distribution.png',
        tmp_path / 'sensor_variance.png',
        tmp_path / 'corr.png',
        tmp_path / 'engine_lifecycle.png',
        tmp_path / 'cycle_distribution.png',
        tmp_path / 'pca.png',
        tmp_path / 'time_series.png',
    ]
    for path in output_files:
        assert path.exists()


def test_identify_columns_and_pca() -> None:
    df = _make_sample_frame()
    constant_info = identify_constant_and_near_constant_columns(df)
    corr = compute_correlation_matrix(df)
    pca = compute_pca_projection(df)
    assert isinstance(constant_info, dict)
    assert corr.shape[0] == corr.shape[1]
    assert pca['components'].shape[1] == 2
    assert engine_lifecycle_summary(df)['max_cycle'].max() == 5
