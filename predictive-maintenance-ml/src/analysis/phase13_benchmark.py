from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.data.load_data import load_cmapss_dataset
from src.evaluation.comparison import compare_model_results
from src.features.target_engineering import compute_training_rul
from src.models.baseline import run_baseline_model_suite, select_feature_columns
from src.utils.config import PROJECT_ROOT


def run_grouped_benchmark(
    train: pd.DataFrame,
    dataset_id: str,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Run the reproducible raw-sensor benchmark and return JSON-safe results."""
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2.')
    feature_columns = select_feature_columns(train, target_column='rul')
    results = run_baseline_model_suite(
        train,
        target_column='rul',
        feature_columns=feature_columns,
        n_splits=n_splits,
        group_column='engine_id',
        include_models=['linear_regression', 'ridge', 'random_forest'],
    )
    comparison = compare_model_results(results, rank_by='rmse')

    model_results = []
    for row in comparison.to_dict(orient='records'):
        model_name = str(row['model'])
        result = results[model_name]
        model_results.append({
            'rank': int(row['rank']),
            'model': model_name,
            'pooled_out_of_fold_metrics': {
                metric: float(row[metric])
                for metric in ('mae', 'rmse', 'r2', 'nasa_score')
            },
            'fold_metric_mean': {
                metric: float(np.mean([fold[metric] for fold in result.fold_metrics]))
                for metric in ('mae', 'rmse', 'r2')
            },
            'fold_metric_std': {
                metric: float(np.std([fold[metric] for fold in result.fold_metrics], ddof=0))
                for metric in ('mae', 'rmse', 'r2')
            },
            'fold_metrics': result.fold_metrics,
        })

    return {
        'dataset': dataset_id,
        'rows': int(len(train)),
        'engines': int(train['engine_id'].nunique()),
        'features': feature_columns,
        'target': 'rul = per-engine maximum training cycle - current cycle',
        'validation': {
            'strategy': 'GroupKFold',
            'group_column': 'engine_id',
            'folds': min(n_splits, int(train['engine_id'].nunique())),
            'ranking_metric': 'pooled out-of-fold RMSE',
        },
        'models': model_results,
        'software': {
            'python': '.'.join(map(str, sys.version_info[:3])),
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'scikit_learn': sklearn.__version__,
        },
    }


def run_local_benchmark(
    dataset_id: str = 'FD001',
    raw_dir: str | Path = PROJECT_ROOT / 'data' / 'raw' / 'CMAPSSData',
    n_splits: int = 3,
) -> dict[str, Any]:
    """Load a local C-MAPSS training file and evaluate it without downloading data."""
    if dataset_id not in {'FD001', 'FD002', 'FD003', 'FD004'}:
        raise ValueError(f'Unsupported C-MAPSS subset: {dataset_id!r}.')
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2.')
    train_path = Path(raw_dir) / f'train_{dataset_id}.txt'
    if not train_path.is_file():
        raise FileNotFoundError(f'Local training file not found: {train_path}.')

    train = load_cmapss_dataset(train_path, dataset_label=dataset_id)
    train['rul'] = compute_training_rul(train)
    return run_grouped_benchmark(train, dataset_id, n_splits=n_splits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run reproducible engine-grouped raw-sensor C-MAPSS baselines.'
    )
    parser.add_argument('--dataset', choices=['FD001', 'FD002', 'FD003', 'FD004'], default='FD001')
    parser.add_argument('--raw-dir', type=Path, default=PROJECT_ROOT / 'data' / 'raw' / 'CMAPSSData')
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'reports' / 'research_paper' / 'fd001_raw_sensor_grouped_cv.json',
    )
    args = parser.parse_args()

    report = run_local_benchmark(args.dataset, args.raw_dir, n_splits=args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Saved {args.dataset} grouped-CV results to {args.output}")
    for model in report['models']:
        metrics = model['pooled_out_of_fold_metrics']
        print(
            f"{model['rank']}. {model['model']}: "
            f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
            f"R2={metrics['r2']:.3f}, NASA score={metrics['nasa_score']:.3g}"
        )


if __name__ == '__main__':
    main()
