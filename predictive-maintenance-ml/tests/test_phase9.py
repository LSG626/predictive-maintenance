from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.comparison import compare_model_results, plot_prediction_diagnostics
from src.evaluation.metrics import nasa_rul_score, regression_metrics
from src.models.baseline import BaselineModelResult


def _result(name: str, predictions: list[float]) -> BaselineModelResult:
    return BaselineModelResult(
        model_name=name,
        fold_metrics=[
            {'mae': 1.0, 'rmse': 1.2, 'r2': 0.8},
            {'mae': 2.0, 'rmse': 2.2, 'r2': 0.7},
        ],
        mean_mae=1.5,
        mean_rmse=1.7,
        mean_r2=0.75,
        predictions=predictions,
        true_values=[10.0, 20.0, 30.0],
    )


def test_nasa_rul_score_uses_asymmetric_penalties() -> None:
    score = nasa_rul_score(np.array([10.0, 10.0]), np.array([0.0, 20.0]))
    expected = np.expm1(10.0 / 13.0) + np.expm1(10.0 / 10.0)
    assert score == pytest.approx(expected)
    assert regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))['nasa_score'] == 0.0


def test_model_comparison_ranks_pooled_predictions_and_reports_fold_spread() -> None:
    results = {
        'accurate': _result('accurate', [10.5, 20.5, 30.5]),
        'inaccurate': _result('inaccurate', [5.0, 25.0, 40.0]),
    }
    comparison = compare_model_results(results, rank_by='mae')

    assert comparison['model'].tolist() == ['accurate', 'inaccurate']
    assert comparison['rank'].tolist() == [1, 2]
    assert comparison.loc[0, 'mae'] == pytest.approx(0.5)
    assert comparison.loc[0, 'mae_std'] == pytest.approx(0.5)
    assert 'nasa_score' in comparison.columns


def test_model_comparison_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match='At least one'):
        compare_model_results({})
    with pytest.raises(ValueError, match='rank_by'):
        compare_model_results({'model': _result('model', [10.0, 20.0, 30.0])}, rank_by='accuracy')
    with pytest.raises(ValueError, match='equally sized'):
        compare_model_results({'bad': _result('bad', [1.0])})


def test_prediction_diagnostics_are_saved(tmp_path) -> None:
    output = plot_prediction_diagnostics(
        np.array([5.0, 10.0, 15.0]),
        np.array([6.0, 9.0, 14.0]),
        tmp_path / 'diagnostics.png',
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_prediction_diagnostics_reject_misaligned_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match='same length'):
        plot_prediction_diagnostics([1.0, 2.0], [1.0], tmp_path / 'bad.png')
