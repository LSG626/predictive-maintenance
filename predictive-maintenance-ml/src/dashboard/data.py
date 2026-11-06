from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import load_cmapss_dataset
from src.features.target_engineering import add_test_rul_column, compute_training_rul


def discover_available_dataset_ids(raw_dir: str | Path) -> list[str]:
    """Return FD subsets with all three required files already available locally."""
    data_dir = Path(raw_dir)
    if not data_dir.exists():
        return []
    dataset_ids = []
    for dataset_id in ('FD001', 'FD002', 'FD003', 'FD004'):
        expected = (
            data_dir / f'train_{dataset_id}.txt',
            data_dir / f'test_{dataset_id}.txt',
            data_dir / f'RUL_{dataset_id}.txt',
        )
        if all(path.is_file() for path in expected):
            dataset_ids.append(dataset_id)
    return dataset_ids


def load_dashboard_dataset(fd_id: str, raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load a local C-MAPSS subset and attach its correctly aligned RUL labels."""
    if fd_id not in {'FD001', 'FD002', 'FD003', 'FD004'}:
        raise ValueError(f'Unsupported C-MAPSS subset: {fd_id!r}.')

    data_dir = Path(raw_dir)
    train_path = data_dir / f'train_{fd_id}.txt'
    test_path = data_dir / f'test_{fd_id}.txt'
    rul_path = data_dir / f'RUL_{fd_id}.txt'
    missing = [path.name for path in (train_path, test_path, rul_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing local dataset files for {fd_id}: {missing}.')

    train = load_cmapss_dataset(train_path, dataset_label=fd_id)
    test = load_cmapss_dataset(test_path, dataset_label=fd_id)
    true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, engine='python')[0]
    train['rul'] = compute_training_rul(train).to_numpy()
    test = add_test_rul_column(test, true_rul.tolist())
    return {'train': train, 'test': test, 'true_rul': true_rul.rename('rul_true').to_frame()}
