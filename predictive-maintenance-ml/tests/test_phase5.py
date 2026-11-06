from __future__ import annotations

import pandas as pd

from src.models.baseline import run_baseline_model_suite


def _build_baseline_dataset() -> pd.DataFrame:
    rows = []
    for engine in [1, 2, 3, 4]:
        for cycle in range(1, 8):
            row = {
                'engine_id': engine,
                'cycle': cycle,
                'rul': 30 - cycle + engine * 0.5,
                'setting_1': 0.5 + cycle,
                'setting_2': 1.0 + cycle * 0.2,
                'setting_3': 1.5 + cycle * 0.1,
            }
            for sensor in range(1, 6):
                row[f'sensor_{sensor}'] = float(sensor * 10 + cycle * 4 + engine * 2)
            rows.append(row)
    return pd.DataFrame(rows)


def test_baseline_model_suite_runs() -> None:
    df = _build_baseline_dataset()
    results = run_baseline_model_suite(
        df,
        target_column='rul',
        n_splits=3,
        include_models=['linear_regression', 'ridge', 'random_forest'],
    )

    assert set(results) == {'linear_regression', 'ridge', 'random_forest'}
    for model_name, result in results.items():
        assert 'mae' in result.fold_metrics[0]
        assert result.mean_r2 <= 1.0
        assert result.mean_r2 > -10.0
        assert len(result.predictions) > 0


def test_selected_feature_columns_exclude_target() -> None:
    df = _build_baseline_dataset()
    feature_columns = [column for column in df.columns if column not in {'engine_id', 'cycle', 'rul'}]
    assert 'rul' not in feature_columns
    assert 'sensor_1' in feature_columns
