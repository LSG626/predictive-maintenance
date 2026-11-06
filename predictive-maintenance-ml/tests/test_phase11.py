from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.tracking.experiment_tracking import log_experiment


class _FakeRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id='test-run-id')

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _fake_mlflow(calls: dict[str, Any]) -> ModuleType:
    module = ModuleType('mlflow')
    module.set_tracking_uri = lambda uri: calls.update(tracking_uri=uri)
    module.set_experiment = lambda name: calls.update(experiment_name=name)
    module.start_run = lambda run_name: (calls.update(run_name=run_name) or _FakeRun())
    module.log_params = lambda params: calls.update(params=params)
    module.log_metrics = lambda metrics: calls.update(metrics=metrics)
    module.set_tags = lambda tags: calls.update(tags=tags)
    module.log_artifact = lambda path, artifact_path=None: calls.setdefault('files', []).append(
        (path, artifact_path)
    )
    module.log_artifacts = lambda path, artifact_path=None: calls.setdefault('directories', []).append(
        (path, artifact_path)
    )
    module.sklearn = SimpleNamespace(
        log_model=lambda model, artifact_path: calls.update(model=(model, artifact_path))
    )
    return module


def test_log_experiment_sends_parameters_metrics_artifacts_and_model(monkeypatch, tmp_path) -> None:
    calls: dict[str, Any] = {}
    artifact = tmp_path / 'trials.csv'
    artifact.write_text('trial,value\n0,1.0\n', encoding='utf-8')
    model = object()
    monkeypatch.setitem(sys.modules, 'mlflow', _fake_mlflow(calls))

    run_id = log_experiment(
        experiment_name='cmapss-test',
        run_name='ridge-run',
        params={'alpha': 1.0, 'options': {'scale': True}},
        metrics={'rmse': 2.5},
        tags={'dataset': 'FD001'},
        artifacts=[artifact],
        model=model,
        tracking_uri='file:./mlruns',
    )

    assert run_id == 'test-run-id'
    assert calls['tracking_uri'] == 'file:./mlruns'
    assert calls['experiment_name'] == 'cmapss-test'
    assert calls['params']['options'] == '{"scale": true}'
    assert calls['metrics'] == {'rmse': 2.5}
    assert calls['tags'] == {'dataset': 'FD001'}
    assert calls['files'] == [(str(artifact), tmp_path.name)]
    assert calls['model'] == (model, 'model')


def test_log_experiment_validates_metrics_and_artifacts_before_tracking(tmp_path) -> None:
    with pytest.raises(ValueError, match='finite'):
        log_experiment('test', 'run', {}, {'rmse': float('nan')})
    with pytest.raises(FileNotFoundError, match='do not exist'):
        log_experiment('test', 'run', {}, {}, artifacts=[tmp_path / 'missing.csv'])


def test_log_experiment_requires_nonempty_names() -> None:
    with pytest.raises(ValueError, match='experiment_name'):
        log_experiment(' ', 'run', {}, {})
    with pytest.raises(ValueError, match='run_name'):
        log_experiment('test', '', {}, {})
