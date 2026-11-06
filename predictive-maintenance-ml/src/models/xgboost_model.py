from __future__ import annotations

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    XGBRegressor = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def build_xgboost_model(**kwargs: object):
    if XGBRegressor is None:
        raise ImportError('xgboost is not installed. Install it to use the XGBoost baseline model.') from _IMPORT_ERROR
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'objective': 'reg:squarederror',
    }
    params.update(kwargs)
    return XGBRegressor(**params)
