from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import (
    build_feature_matrix,
    select_features_by_correlation,
    select_features_by_mutual_information,
    select_features_by_rfe,
    select_features_by_variance_threshold,
)


def _make_training_frame() -> pd.DataFrame:
    rows = []
    for engine in [1, 2, 3]:
        for cycle in range(1, 6):
            row = {'engine_id': engine, 'cycle': cycle, 'rul': 10 - cycle if cycle <= 10 else 0}
            for sensor in range(1, 5):
                row[f'sensor_{sensor}'] = float(sensor * 10 + engine * 2 + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_feature_matrix_includes_rolling_and_diff_features() -> None:
    df = _make_training_frame()
    features = build_feature_matrix(df, target_column='rul', windows=(3, 5))
    assert 'sensor_1_roll_mean_3' in features.columns
    assert 'sensor_1_diff_1' in features.columns
    assert 'sensor_1_ewm' in features.columns
    assert 'rul' in features.columns
    assert features.shape[0] == df.shape[0]


def test_feature_selection_methods_return_values() -> None:
    df = _make_training_frame()
    features = build_feature_matrix(df, target_column='rul', windows=(3, 5))

    variance_selected = select_features_by_variance_threshold(features, target_column='rul')
    corr_selected = select_features_by_correlation(features, target_column='rul')
    mi_selected = select_features_by_mutual_information(features, target_column='rul', top_k=5)
    rfe_selected = select_features_by_rfe(features, target_column='rul', n_features=5)

    assert len(variance_selected) > 0
    assert len(corr_selected) > 0
    assert len(mi_selected) > 0
    assert len(rfe_selected) > 0


def test_engineered_features_are_past_only() -> None:
    df = _make_training_frame()
    features = build_feature_matrix(df, target_column='rul', windows=(3, 5))
    engine_1 = features[features['engine_id'] == 1]
    assert engine_1['sensor_1_roll_mean_3'].iloc[0] == 0.0
    assert engine_1['sensor_1_diff_1'].iloc[0] == 0.0
