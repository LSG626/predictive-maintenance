from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _validate_features(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame with named feature columns.')
    if X.empty or X.shape[1] == 0:
        raise ValueError('X must contain at least one row and one feature.')
    if not X.columns.is_unique:
        raise ValueError('X must have unique feature column names.')


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    scoring: str = 'neg_mean_absolute_error',
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Measure holdout feature importance by the score decrease after permutation."""
    _validate_features(X)
    if len(y) != len(X):
        raise ValueError('X and y must contain the same number of rows.')
    if n_repeats < 1:
        raise ValueError('n_repeats must be at least 1.')

    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    })
    return importance.sort_values('importance_mean', ascending=False, ignore_index=True)


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute global mean-absolute SHAP values for a bounded, reproducible sample."""
    _validate_features(X)
    if max_samples < 1:
        raise ValueError('max_samples must be at least 1.')

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            'SHAP explainability requires the optional "shap" dependency. '
            'Install the project dependencies in a supported Python environment.'
        ) from exc

    sample = X
    if len(X) > max_samples:
        sample = X.sample(n=max_samples, random_state=random_state)
    explanation = shap.Explainer(model, sample)(sample)
    values = np.asarray(explanation.values)
    if values.ndim < 2:
        raise ValueError('SHAP returned values without a sample and feature dimension.')

    feature_axis = 1 if values.shape[1] == X.shape[1] else values.ndim - 1
    if values.shape[feature_axis] != X.shape[1]:
        raise ValueError('SHAP output feature dimension does not match X.')
    reduce_axes = tuple(axis for axis in range(values.ndim) if axis != feature_axis)
    absolute_values = np.abs(values)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': absolute_values.mean(axis=reduce_axes),
        'importance_std': absolute_values.std(axis=reduce_axes),
    })
    return importance.sort_values('importance_mean', ascending=False, ignore_index=True)


def plot_feature_importance(
    importance: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 20,
    title: str = 'Global feature importance',
) -> Path:
    """Save a horizontal bar chart from either supported importance result."""
    required = {'feature', 'importance_mean'}
    if not required.issubset(importance.columns):
        raise ValueError(f'importance must contain columns {sorted(required)}.')
    if top_n < 1:
        raise ValueError('top_n must be at least 1.')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    displayed = importance.nlargest(top_n, 'importance_mean').sort_values('importance_mean')
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(displayed))))
    sns.barplot(
        data=displayed,
        x='importance_mean',
        y='feature',
        hue='feature',
        dodge=False,
        legend=False,
        palette='viridis',
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel('Mean importance')
    ax.set_ylabel('Feature')
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path
