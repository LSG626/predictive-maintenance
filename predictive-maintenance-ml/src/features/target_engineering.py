from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def compute_training_rul(df: pd.DataFrame, threshold: int | None = None) -> pd.Series:
    """Compute uncapped and optionally capped RUL labels for training data.

    The NASA C-MAPSS convention is:
      RUL = max_observed_cycle_for_engine - current_cycle
    """
    if 'engine_id' not in df.columns or 'cycle' not in df.columns:
        raise ValueError('Training DataFrame must contain engine_id and cycle columns.')

    max_cycle_by_engine = df.groupby('engine_id')['cycle'].transform('max')
    rul = max_cycle_by_engine - df['cycle']
    if threshold is not None:
        rul = rul.clip(lower=0, upper=threshold)
    return pd.Series(rul, index=df.index, name='rul')


def compute_capped_rul_thresholds(df: pd.DataFrame, candidate_thresholds: Sequence[int] | None = None) -> dict[int, pd.Series]:
    """Create RUL targets for multiple candidate capping thresholds."""
    if candidate_thresholds is None:
        candidate_thresholds = [60, 80, 100, 125, 150, 200]

    threshold_map: dict[int, pd.Series] = {}
    uncapped_rul = compute_training_rul(df, threshold=None)
    for threshold in candidate_thresholds:
        threshold_map[int(threshold)] = uncapped_rul.clip(lower=0, upper=int(threshold)).rename(f'rul_capped_{threshold}')
    return threshold_map


def add_training_rul_columns(df: pd.DataFrame, candidate_thresholds: Sequence[int] | None = None) -> pd.DataFrame:
    """Append uncapped and capped training targets to a C-MAPSS frame."""
    out = df.copy()
    uncapped = compute_training_rul(out, threshold=None)
    out['rul_uncapped'] = uncapped.to_numpy()
    for threshold, series in compute_capped_rul_thresholds(out, candidate_thresholds).items():
        out[f'rul_capped_{threshold}'] = series.to_numpy()
    return out


def add_test_rul_column(test_df: pd.DataFrame, true_rul_values: Sequence[int], engine_col: str = 'engine_id', cycle_col: str = 'cycle') -> pd.DataFrame:
    """Construct aligned RUL labels for the test set using the supplied ground-truth file.

    NASA C-MAPSS test labels are not a row-wise constant; they represent the remaining useful life
    reported at the final observed cycle of each engine. For a given row at cycle c and final cycle F,
    the appropriate target is:
        true_rul_at_last_cycle - (F - c)
    """
    if engine_col not in test_df.columns or cycle_col not in test_df.columns:
        raise ValueError('Test DataFrame must contain the engine and cycle columns.')

    if len(true_rul_values) != test_df[engine_col].nunique():
        raise ValueError(
            f'Expected {test_df[engine_col].nunique()} RUL values for the test set, '
            f'but received {len(true_rul_values)}.'
        )

    final_cycle_by_engine = test_df.groupby(engine_col)[cycle_col].max().rename('final_cycle')
    rul_lookup = pd.Series(list(true_rul_values), index=sorted(test_df[engine_col].unique()), name='rul_at_final_cycle')
    engine_labels = final_cycle_by_engine.to_frame().join(rul_lookup, how='left')
    labeled = test_df.merge(engine_labels, left_on=engine_col, right_index=True, how='left')
    labeled['rul'] = labeled['rul_at_final_cycle'] + labeled['final_cycle'] - labeled[cycle_col]
    return labeled.drop(columns=['rul_at_final_cycle'])


def prepare_fd_dataset(fd_id: str, raw_dir: str | Path = 'data/raw/CMAPSSData', threshold: int | None = None) -> dict[str, pd.DataFrame]:
    """Load a single FD dataset, add training labels and test labels, and return both DataFrames."""
    raw_dir = Path(raw_dir)
    train_path = raw_dir / f'train_{fd_id}.txt'
    test_path = raw_dir / f'test_{fd_id}.txt'
    rul_path = raw_dir / f'RUL_{fd_id}.txt'

    if not train_path.exists() or not test_path.exists() or not rul_path.exists():
        raise FileNotFoundError(f'Incomplete C-MAPSS files for {fd_id}. Expected {train_path.name}, {test_path.name}, and {rul_path.name}.')

    columns = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, engine='python', names=columns)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, engine='python', names=columns)
    true_rul_values = pd.read_csv(rul_path, sep=r'\s+', header=None, engine='python')[0].tolist()

    train_labeled = add_training_rul_columns(train_df, candidate_thresholds=[60, 80, 100, 125, 150, 200])
    if threshold is not None:
        train_labeled['rul'] = train_labeled['rul_uncapped'].clip(lower=0, upper=int(threshold))
    test_labeled = add_test_rul_column(test_df, true_rul_values)
    return {'train': train_labeled, 'test': test_labeled, 'true_rul': pd.Series(true_rul_values, name='rul_true')}
