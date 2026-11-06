from __future__ import annotations

import pytest

from src.analysis.phase14_cross_subset_benchmark import (
    run_cross_subset_benchmark,
    summarize_rank_consistency,
)


def _subset_result(dataset: str, ranks: tuple[int, int, int]) -> dict:
    names = ('linear_regression', 'ridge', 'random_forest')
    return {
        'dataset': dataset,
        'models': [
            {'model': name, 'rank': rank}
            for name, rank in zip(names, ranks)
        ],
    }


def test_rank_summary_aggregates_ranks_not_metrics() -> None:
    summary = summarize_rank_consistency({
        'FD001': _subset_result('FD001', (3, 2, 1)),
        'FD002': _subset_result('FD002', (2, 3, 1)),
        'FD003': _subset_result('FD003', (1, 2, 3)),
    })

    assert summary[0] == {
        'model': 'random_forest',
        'mean_rank': 5 / 3,
        'rank_1_count': 2,
        'subsets_evaluated': 3,
    }
    assert summary[1]['model'] == 'linear_regression'
    assert summary[1]['mean_rank'] == pytest.approx(2.0)
    assert summary[2]['model'] == 'ridge'


def test_rank_summary_rejects_missing_models_and_empty_results() -> None:
    with pytest.raises(ValueError, match='At least one'):
        summarize_rank_consistency({})
    with pytest.raises(ValueError, match='no model results'):
        summarize_rank_consistency({'FD001': {'models': []}})
    with pytest.raises(ValueError, match='present in every subset'):
        summarize_rank_consistency({
            'FD001': _subset_result('FD001', (1, 2, 3)),
            'FD002': {'models': [{'model': 'ridge', 'rank': 1}]},
        })


def test_cross_subset_benchmark_validates_selection_before_running(tmp_path) -> None:
    with pytest.raises(ValueError, match='n_splits'):
        run_cross_subset_benchmark(tmp_path, ['FD001'], n_splits=1)
    with pytest.raises(ValueError, match='duplicates'):
        run_cross_subset_benchmark(tmp_path, ['FD001', 'FD001'])
    with pytest.raises(ValueError, match='Unsupported'):
        run_cross_subset_benchmark(tmp_path, ['FD999'])
    with pytest.raises(FileNotFoundError, match='Missing local training files'):
        run_cross_subset_benchmark(tmp_path, ['FD001'])
