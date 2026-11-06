from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import RegressorMixin

from src.models.hyperparameter_optimization import optimize_hyperparameters


def _make_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for engine_id in range(6):
        for cycle in range(8):
            signal = float(cycle + rng.normal(0, 0.05))
            rows.append({
                'engine_id': engine_id,
                'cycle': cycle,
                'signal': signal,
                'noise': float(rng.normal()),
                'rul': 30.0 - 2.0 * signal,
            })
    return pd.DataFrame(rows)


def test_optimization_uses_grouped_cv_and_returns_fitted_best_model() -> None:
    result = optimize_hyperparameters(
        _make_dataset(),
        model_name='ridge',
        n_splits=3,
        n_trials=3,
        random_state=7,
    )

    assert result.model_name == 'ridge'
    assert result.metric == 'rmse'
    assert result.best_value >= 0
    assert result.trials.shape[0] == 3
    assert {'trial', 'value', 'state', 'alpha'}.issubset(result.trials.columns)
    assert isinstance(result.best_model, RegressorMixin)
    assert result.best_model.predict(pd.DataFrame({'signal': [1.0], 'noise': [0.0]})).shape == (1,)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'model_name': 'linear_regression'}, 'model_name'),
        ({'metric': 'accuracy'}, 'metric'),
        ({'n_trials': 0}, 'n_trials'),
        ({'n_splits': 1}, 'n_splits'),
        ({'feature_columns': []}, 'feature column'),
    ],
)
def test_optimization_rejects_invalid_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        optimize_hyperparameters(_make_dataset(), **kwargs)


def test_optimization_requires_two_distinct_engines() -> None:
    df = _make_dataset()
    df['engine_id'] = 1
    with pytest.raises(ValueError, match='two unique groups'):
        optimize_hyperparameters(df, model_name='ridge', n_trials=1)
