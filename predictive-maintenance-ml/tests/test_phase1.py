from __future__ import annotations

import pandas as pd

from src.data.validate_data import validate_dataset


def _make_sample_frame() -> pd.DataFrame:
    rows = []
    for engine in [1, 2]:
        for cycle in [1, 2, 3]:
            row = {
                'engine_id': engine,
                'cycle': cycle,
                'setting_1': 0.5,
                'setting_2': 0.4,
                'setting_3': 0.3,
            }
            for sensor in range(1, 22):
                row[f'sensor_{sensor}'] = float(sensor + engine + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


def test_validate_dataset() -> None:
    df = _make_sample_frame()
    report = validate_dataset(df)

    assert report['n_rows'] == 6
    assert report['n_engines'] == 2
    assert report['duplicate_rows'] == 0
    assert report['missing_values_total'] == 0
    assert report['min_cycle'] == 1
    assert report['max_cycle'] == 3


def test_validate_dataset_raises_on_missing_columns() -> None:
    df = _make_sample_frame().drop(columns=['sensor_21'])
    try:
        validate_dataset(df)
        raise AssertionError('Expected a ValueError for missing columns')
    except ValueError:
        pass
