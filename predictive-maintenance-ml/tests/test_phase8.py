from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from src.explainability.feature_importance import (
    compute_permutation_importance,
    compute_shap_importance,
    plot_feature_importance,
)


def _make_explainability_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    n_engines = 8
    cycles = 12
    engine_id = np.repeat(np.arange(n_engines), cycles)
    cycle = np.tile(np.arange(cycles), n_engines)
    signal = cycle + rng.normal(0, 0.1, len(cycle))
    noise = rng.normal(0, 1, len(cycle))
    X = pd.DataFrame({'degradation_signal': signal, 'noise_sensor': noise})
    y = pd.Series(100 - 3 * signal + rng.normal(0, 0.1, len(signal)))
    return X, y, pd.Series(engine_id)


def test_permutation_importance_ranks_predictive_feature_and_plots(tmp_path) -> None:
    X, y, groups = _make_explainability_dataset()
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups)
    )
    model = RandomForestRegressor(n_estimators=40, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    importance = compute_permutation_importance(
        model, X.iloc[test_idx], y.iloc[test_idx], n_repeats=3, random_state=42
    )
    assert importance.iloc[0]['feature'] == 'degradation_signal'
    assert set(importance.columns) == {'feature', 'importance_mean', 'importance_std'}

    output = plot_feature_importance(importance, tmp_path / 'importance.png', top_n=2)
    assert output.exists()
    assert output.stat().st_size > 0


def test_permutation_importance_rejects_misaligned_inputs() -> None:
    X, y, _ = _make_explainability_dataset()
    model = RandomForestRegressor(n_estimators=5, random_state=42).fit(X, y)
    with pytest.raises(ValueError, match='same number of rows'):
        compute_permutation_importance(model, X, y.iloc[:-1])


def test_shap_importance_reports_clear_optional_dependency_error() -> None:
    X, _, _ = _make_explainability_dataset()
    model = RandomForestRegressor(n_estimators=5, random_state=42).fit(X, np.arange(len(X)))
    if importlib.util.find_spec('shap') is None:
        with pytest.raises(ImportError, match='optional "shap" dependency'):
            compute_shap_importance(model, X, max_samples=10)
        return
    importance = compute_shap_importance(model, X, max_samples=10)
    assert importance.iloc[0]['importance_mean'] >= importance.iloc[-1]['importance_mean']
