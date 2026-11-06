from __future__ import annotations

import pandas as pd

from src.models.anomaly_detection import (
    build_anomaly_summary,
    detect_isolation_forest,
    detect_lof,
    detect_one_class_svm,
)


def _make_anomaly_dataset() -> pd.DataFrame:
    rows = []
    for engine in [1, 2, 3]:
        for cycle in range(1, 11):
            row = {
                'engine_id': engine,
                'cycle': cycle,
                'setting_1': 1.0 + cycle * 0.1,
                'setting_2': 2.0 + cycle * 0.2,
                'setting_3': 3.0 + cycle * 0.3,
            }
            for sensor in range(1, 6):
                row[f'sensor_{sensor}'] = float(sensor * 10 + cycle * 3 + engine)
            rows.append(row)
    rows.append({
        'engine_id': 4,
        'cycle': 1,
        'setting_1': 100.0,
        'setting_2': 100.0,
        'setting_3': 100.0,
        'sensor_1': 100.0,
        'sensor_2': 100.0,
        'sensor_3': 100.0,
        'sensor_4': 100.0,
        'sensor_5': 100.0,
    })
    return pd.DataFrame(rows)


def test_anomaly_models_build_and_score() -> None:
    df = _make_anomaly_dataset()
    iso = detect_isolation_forest(df)
    svm = detect_one_class_svm(df)
    lof = detect_lof(df)

    iso_summary = build_anomaly_summary(df, iso)
    svm_summary = build_anomaly_summary(df, svm)
    lof_summary = build_anomaly_summary(df, lof)

    assert 'anomaly_score' in iso_summary.columns
    assert 'is_anomaly' in iso_summary.columns
    assert 'anomaly_score' in svm_summary.columns
    assert 'is_anomaly' in svm_summary.columns
    assert 'anomaly_score' in lof_summary.columns
    assert 'is_anomaly' in lof_summary.columns
    assert iso_summary['anomaly_score'].notna().all()
    assert svm_summary['anomaly_score'].notna().all()
    assert lof_summary['anomaly_score'].notna().all()


def test_anomaly_detection_uses_numeric_features_only() -> None:
    df = _make_anomaly_dataset()
    iso = detect_isolation_forest(df)
    summary = build_anomaly_summary(df, iso)
    assert 'engine_id' in summary.columns
    assert 'anomaly_score' in summary.columns
    assert set(summary.columns).issuperset({'engine_id', 'cycle', 'setting_1', 'sensor_1'})
