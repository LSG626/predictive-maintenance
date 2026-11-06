from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def generate_sequences(
    df: pd.DataFrame,
    sequence_length: int = 30,
    target_column: str = 'rul',
    group_column: str = 'engine_id',
    feature_columns: Sequence[str] | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed-length supervised sequences from a time-series DataFrame.

    The function preserves engine grouping to avoid mixing trajectories between engines and uses
    only prior observations within each engine. This is the standard pattern for LSTM/GRU/CNN
    sequence models.
    """
    if sequence_length <= 0:
        raise ValueError('sequence_length must be a positive integer.')
    if stride <= 0:
        raise ValueError('stride must be a positive integer.')

    if feature_columns is None:
        feature_columns = [
            column for column in df.columns
            if column not in {group_column, 'cycle', 'dataset', target_column}
            and pd.api.types.is_numeric_dtype(df[column])
        ]

    if not feature_columns:
        raise ValueError('No numeric feature columns found to build sequences.')

    X_frames: list[np.ndarray] = []
    y_frames: list[np.ndarray] = []

    for _, engine_data in df.sort_values([group_column, 'cycle']).groupby(group_column, sort=True):
        engine_features = engine_data[feature_columns].to_numpy(dtype=float)
        engine_target = engine_data[target_column].to_numpy(dtype=float)

        if len(engine_features) < sequence_length:
            continue

        for start_idx in range(0, len(engine_features) - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            sequence = engine_features[start_idx:end_idx]
            target = engine_target[end_idx - 1]
            X_frames.append(sequence)
            y_frames.append(target)

    if not X_frames:
        raise ValueError('No valid sequences were generated. Check the min engine lifetime and sequence length.')

    return np.asarray(X_frames, dtype=float), np.asarray(y_frames, dtype=float)
