"""MLflow client runtime configuration shared by the API and the Celery worker."""

from __future__ import annotations

import os

import mlflow

from kepler_engine.core.config import Settings


def configure_mlflow_runtime(settings: Settings) -> None:
    """Set the tracking URI and bound MLflow's HTTP retries.

    MLflow's default ladder is 7 retries with exponential backoff, so one call against
    an unreachable tracking server blocks for minutes and stalls both startup and the
    readiness probe. Explicitly set MLFLOW_* variables still win.

    Every process that talks to the tracking server must call this. The worker is the
    process that needs it most: a training run issues far more REST calls than a
    readiness probe, so an artifact-store hiccup on the default ladder silently pins a
    worker slot for minutes per call instead of failing the job fast.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", str(settings.mlflow_http_max_retries))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(settings.mlflow_http_timeout_seconds))
    os.environ.setdefault(
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", str(settings.mlflow_http_backoff_factor)
    )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
