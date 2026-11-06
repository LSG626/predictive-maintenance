from __future__ import annotations

from sklearn.svm import SVR


def build_svr_model(**kwargs: object) -> SVR:
    params = {
        'kernel': 'rbf',
        'C': 100.0,
        'epsilon': 0.1,
        'gamma': 'scale',
    }
    params.update(kwargs)
    return SVR(**params)
