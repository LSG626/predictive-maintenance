from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard.data import discover_available_dataset_ids, load_dashboard_dataset
from src.data.load_data import DEFAULT_COLUMN_NAMES


def _write_subset(raw_dir, fd_id: str = 'FD001') -> None:
    rows = []
    for engine_id in (1, 2):
        for cycle in range(1, 4):
            values = [engine_id, cycle, 0.1, 0.2, 0.3]
            values.extend(float(engine_id * 10 + cycle + sensor) for sensor in range(1, 22))
            rows.append(values)
    frame = pd.DataFrame(rows, columns=DEFAULT_COLUMN_NAMES)
    frame.to_csv(raw_dir / f'train_{fd_id}.txt', sep=' ', header=False, index=False)
    frame.to_csv(raw_dir / f'test_{fd_id}.txt', sep=' ', header=False, index=False)
    (raw_dir / f'RUL_{fd_id}.txt').write_text('5\n7\n', encoding='utf-8')


def test_discover_available_subsets_requires_complete_file_triplets(tmp_path) -> None:
    _write_subset(tmp_path)
    (tmp_path / 'train_FD002.txt').write_text('partial', encoding='utf-8')
    assert discover_available_dataset_ids(tmp_path) == ['FD001']
    assert discover_available_dataset_ids(tmp_path / 'missing') == []


def test_load_dashboard_data_adds_training_and_test_rul(tmp_path) -> None:
    _write_subset(tmp_path)
    data = load_dashboard_dataset('FD001', tmp_path)

    assert set(data) == {'train', 'test', 'true_rul'}
    assert data['train']['rul'].tolist() == [2, 1, 0, 2, 1, 0]
    assert data['test']['rul'].tolist() == [7, 6, 5, 9, 8, 7]
    assert data['true_rul']['rul_true'].tolist() == [5, 7]


def test_load_dashboard_data_reports_missing_and_invalid_subsets(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match='Missing local dataset files'):
        load_dashboard_dataset('FD001', tmp_path)
    with pytest.raises(ValueError, match='Unsupported'):
        load_dashboard_dataset('FD999', tmp_path)
