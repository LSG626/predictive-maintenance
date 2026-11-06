from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def build_random_forest_model(**kwargs: object) -> RandomForestRegressor:
    params = {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_leaf': 2,
        'random_state': 42,
    }
    params.update(kwargs)
    return RandomForestRegressor(**params)
