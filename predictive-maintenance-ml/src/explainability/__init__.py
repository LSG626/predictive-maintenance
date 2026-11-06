"""Model explainability utilities."""

from src.explainability.feature_importance import (
    compute_permutation_importance,
    compute_shap_importance,
    plot_feature_importance,
)

__all__ = [
    'compute_permutation_importance',
    'compute_shap_importance',
    'plot_feature_importance',
]
