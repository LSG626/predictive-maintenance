from __future__ import annotations

import pandas as pd

from src.features.target_engineering import (
    add_test_rul_column,
    add_training_rul_columns,
    compute_training_rul,
    prepare_fd_dataset,
)


def _build_training_frame() -> pd.DataFrame:
    rows = []
    for engine in [1, 2]:
        for cycle in [1, 2, 3, 4]:
            row = {'engine_id': engine, 'cycle': cycle, 'setting_1': 1.0, 'setting_2': 2.0, 'setting_3': 3.0}
            for sensor in range(1, 22):
                row[f'sensor_{sensor}'] = float(sensor + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_test_frame() -> pd.DataFrame:
    rows = []
    for engine, final_cycle in [(1, 5), (2, 4)]:
        for cycle in range(1, final_cycle + 1):
            row = {'engine_id': engine, 'cycle': cycle, 'setting_1': 1.0, 'setting_2': 2.0, 'setting_3': 3.0}
            for sensor in range(1, 22):
                row[f'sensor_{sensor}'] = float(sensor + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


def test_training_rul_and_thresholds() -> None:
    df = _build_training_frame()
    rul = compute_training_rul(df)
    assert rul.iloc[0] == 3
    assert rul.max() == 3

    capped = add_training_rul_columns(df, candidate_thresholds=[3, 5])
    assert 'rul_uncapped' in capped.columns
    assert 'rul_capped_3' in capped.columns
    assert capped['rul_capped_3'].max() <= 3
    assert capped['rul_capped_5'].max() <= 5


def test_test_rul_label_construction() -> None:
    test_df = _build_test_frame()
    label_values = [12, 8]
    labeled = add_test_rul_column(test_df, label_values)

    engine1_values = labeled[labeled['engine_id'] == 1]['rul']
    engine2_values = labeled[labeled['engine_id'] == 2]['rul']
    assert engine1_values.iloc[0] == 16
    assert engine1_values.iloc[-1] == 12
    assert engine2_values.iloc[0] == 11
    assert engine2_values.iloc[-1] == 8


def test_prepare_fd001_dataset() -> None:
    dataset = prepare_fd_dataset('FD001', raw_dir='data/raw/CMAPSSData')
    assert not dataset['train'].empty
    assert not dataset['test'].empty
    assert 'rul_uncapped' in dataset['train'].columns
    assert 'rul' in dataset['test'].columns
    assert dataset['train']['engine_id'].nunique() > 1
    assert dataset['test']['engine_id'].nunique() > 1
