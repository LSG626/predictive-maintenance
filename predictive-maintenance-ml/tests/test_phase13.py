from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.phase13_benchmark import run_grouped_benchmark, run_local_benchmark


def _synthetic_training_data() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for engine_id in range(6):
        for cycle in range(1, 6):
            rows.append({
                'engine_id': engine_id,
                'cycle': cycle,
                'setting_1': float(cycle) * 0.1,
                'sensor_1': float(cycle + rng.normal(0, 0.05)),
                'rul': float(6 - cycle),
            })
    return pd.DataFrame(rows)


def test_grouped_benchmark_returns_engine_aware_model_metrics() -> None:
    report = run_grouped_benchmark(_synthetic_training_data(), 'SYNTHETIC', n_splits=3)

    assert report['rows'] == 30
    assert report['engines'] == 6
    assert report['validation']['strategy'] == 'GroupKFold'
    assert report['validation']['folds'] == 3
    assert {item['model'] for item in report['models']} == {
        'random_forest',
        'ridge',
        'linear_regression',
    }
    assert [item['rank'] for item in report['models']] == [1, 2, 3]
    assert all(
        set(item['pooled_out_of_fold_metrics']) == {'mae', 'rmse', 'r2', 'nasa_score'}
        for item in report['models']
    )


def test_local_benchmark_rejects_missing_files_and_invalid_folds(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match='not found'):
        run_local_benchmark('FD001', tmp_path)
    with pytest.raises(ValueError, match='n_splits'):
        run_local_benchmark('FD001', tmp_path, n_splits=1)
