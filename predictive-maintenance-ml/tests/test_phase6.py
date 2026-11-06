from __future__ import annotations

import pandas as pd

from src.features.sequence_generator import generate_sequences
from src.models.deep_learning import run_deep_learning_model_suite


def _make_sequence_dataframe() -> pd.DataFrame:
    rows = []
    for engine in [1, 2, 3]:
        for cycle in range(1, 18):
            row = {'engine_id': engine, 'cycle': cycle, 'rul': 17 - cycle, 'setting_1': 1.0, 'setting_2': 2.0, 'setting_3': 3.0}
            for sensor in range(1, 6):
                row[f'sensor_{sensor}'] = float(sensor + cycle + engine)
            rows.append(row)
    return pd.DataFrame(rows)


def test_sequence_generation_creates_valid_arrays() -> None:
    df = _make_sequence_dataframe()
    X, y = generate_sequences(df, sequence_length=5, target_column='rul')
    assert X.shape[0] == y.shape[0]
    assert X.shape[1:] == (5, 8)
    assert y.ndim == 1
    assert X.dtype.kind == 'f'


def test_deep_learning_suite_builds_models() -> None:
    df = _make_sequence_dataframe()
    models = run_deep_learning_model_suite(df, sequence_length=5, target_column='rul', include_models=['lstm', 'gru', 'cnn'])
    assert set(models) == {'lstm', 'gru', 'cnn'}
    assert models['lstm'].available in {True, False}
    assert isinstance(models['lstm'].sample_predictions, list)
