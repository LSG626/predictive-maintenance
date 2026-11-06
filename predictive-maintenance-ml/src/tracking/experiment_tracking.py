from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _serialize_parameter(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def log_experiment(
    experiment_name: str,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    tags: Mapping[str, Any] | None = None,
    artifacts: Sequence[str | Path] = (),
    model: Any | None = None,
    model_artifact_path: str = 'model',
    tracking_uri: str | None = None,
) -> str:
    """Log a completed experiment to MLflow and return its run ID.

    MLflow is imported lazily so training and evaluation remain usable without
    the optional tracking runtime installed.
    """
    if not experiment_name.strip():
        raise ValueError('experiment_name must not be empty.')
    if not run_name.strip():
        raise ValueError('run_name must not be empty.')
    if not model_artifact_path.strip():
        raise ValueError('model_artifact_path must not be empty.')

    normalized_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f'Metric {key!r} must be finite.')
        normalized_metrics[key] = numeric_value

    artifact_paths = [Path(path) for path in artifacts]
    missing_artifacts = [str(path) for path in artifact_paths if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError(f'Artifact paths do not exist: {missing_artifacts}.')

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            'Experiment tracking requires MLflow. Install the project dependencies '
            'in a supported Python environment (for example, pip install mlflow).'
        ) from exc

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params({key: _serialize_parameter(value) for key, value in params.items()})
        if normalized_metrics:
            mlflow.log_metrics(normalized_metrics)
        if tags:
            mlflow.set_tags({key: _serialize_parameter(value) for key, value in tags.items()})
        for artifact in artifact_paths:
            if artifact.is_dir():
                mlflow.log_artifacts(str(artifact), artifact_path=artifact.name)
            else:
                mlflow.log_artifact(str(artifact), artifact_path=artifact.parent.name or None)
        if model is not None:
            mlflow.sklearn.log_model(model, artifact_path=model_artifact_path)
        return str(active_run.info.run_id)
