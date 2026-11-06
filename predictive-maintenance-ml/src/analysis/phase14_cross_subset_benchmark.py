from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis.phase13_benchmark import run_grouped_benchmark
from src.data.load_data import load_cmapss_dataset
from src.features.target_engineering import compute_training_rul
from src.utils.config import PROJECT_ROOT

SUPPORTED_DATASETS = ('FD001', 'FD002', 'FD003', 'FD004')


def summarize_rank_consistency(
    subset_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, float | int | str]]:
    """Summarize within-subset RMSE ranks without pooling different datasets."""
    if not subset_results:
        raise ValueError('At least one subset result is required.')

    rank_values: dict[str, list[int]] = {}
    win_counts: dict[str, int] = {}
    for dataset_id, result in subset_results.items():
        models = result.get('models', [])
        if not models:
            raise ValueError(f'{dataset_id} has no model results.')
        names = [str(model['model']) for model in models]
        if len(set(names)) != len(names):
            raise ValueError(f'{dataset_id} contains duplicate model results.')
        for model in models:
            rank = int(model['rank'])
            if rank < 1:
                raise ValueError(f'{dataset_id} contains an invalid model rank.')
            name = str(model['model'])
            rank_values.setdefault(name, []).append(rank)
            win_counts.setdefault(name, 0)
            if rank == 1:
                win_counts[name] += 1

    dataset_count = len(subset_results)
    if any(len(ranks) != dataset_count for ranks in rank_values.values()):
        raise ValueError('Every model must be present in every subset to compare rank consistency.')
    return sorted(
        [
            {
                'model': model_name,
                'mean_rank': sum(ranks) / len(ranks),
                'rank_1_count': win_counts[model_name],
                'subsets_evaluated': len(ranks),
            }
            for model_name, ranks in rank_values.items()
        ],
        key=lambda item: (float(item['mean_rank']), str(item['model'])),
    )


def run_cross_subset_benchmark(
    raw_dir: str | Path = PROJECT_ROOT / 'data' / 'raw' / 'CMAPSSData',
    dataset_ids: Sequence[str] = SUPPORTED_DATASETS,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Evaluate the same raw-sensor baselines independently on each subset."""
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2.')
    if not dataset_ids:
        raise ValueError('At least one C-MAPSS subset must be selected.')
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError('dataset_ids must not contain duplicates.')
    unsupported = sorted(set(dataset_ids).difference(SUPPORTED_DATASETS))
    if unsupported:
        raise ValueError(f'Unsupported C-MAPSS subsets: {unsupported}.')

    data_dir = Path(raw_dir)
    missing = [
        str(data_dir / f'train_{dataset_id}.txt')
        for dataset_id in dataset_ids
        if not (data_dir / f'train_{dataset_id}.txt').is_file()
    ]
    if missing:
        raise FileNotFoundError(f'Missing local training files: {missing}.')

    subset_results: dict[str, dict[str, Any]] = {}
    for dataset_id in dataset_ids:
        train = load_cmapss_dataset(
            data_dir / f'train_{dataset_id}.txt',
            dataset_label=dataset_id,
        )
        train['rul'] = compute_training_rul(train)
        subset_results[dataset_id] = run_grouped_benchmark(
            train,
            dataset_id,
            n_splits=n_splits,
        )

    return {
        'protocol': {
            'validation': 'GroupKFold by engine_id, independently within each subset',
            'requested_folds': n_splits,
            'models': ['linear_regression', 'ridge', 'random_forest'],
            'features': '24 raw operating-setting and sensor measurements',
            'target': 'per-engine maximum training cycle minus current cycle',
            'primary_metric': 'pooled out-of-fold RMSE within each subset',
        },
        'interpretation': (
            'Metrics are reported separately because the subsets differ in operating '
            'conditions, fault modes, and sample counts. Mean rank summarizes relative '
            'model ordering across subsets; it is not a pooled performance metric.'
        ),
        'subsets': subset_results,
        'rank_consistency': summarize_rank_consistency(subset_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate raw-sensor RUL baselines independently across C-MAPSS subsets.'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
    )
    parser.add_argument('--raw-dir', type=Path, default=PROJECT_ROOT / 'data' / 'raw' / 'CMAPSSData')
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'reports' / 'research_paper' / 'cross_subset_grouped_cv.json',
    )
    args = parser.parse_args()

    report = run_cross_subset_benchmark(args.raw_dir, args.datasets, args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Saved cross-subset results to {args.output}')
    for result in report['rank_consistency']:
        print(
            f"{result['model']}: mean rank {result['mean_rank']:.2f}, "
            f"rank 1 on {result['rank_1_count']}/{result['subsets_evaluated']} subsets"
        )


if __name__ == '__main__':
    main()
