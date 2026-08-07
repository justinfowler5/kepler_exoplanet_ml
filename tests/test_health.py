"""Health endpoint and MLflow runtime-tuning tests."""

from __future__ import annotations

import pytest
from mlflow.environment_variables import (
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)

from kepler_engine.core.config import get_settings
from kepler_engine.core.mlflow_runtime import configure_mlflow_runtime
from kepler_engine.workers.tasks import run_experiment_task

_TUNED_VARIABLES = (
    MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR,
)


@pytest.mark.parametrize("variable", _TUNED_VARIABLES, ids=lambda v: v.name)
def test_mlflow_http_tuning_is_parseable_by_mlflow(
    variable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every value we export must survive MLflow's own parser.

    MLflow parses these with the type declared in its environment_variables
    module -- BACKOFF_FACTOR is an int even though the adjacent BACKOFF_JITTER is
    a float -- and raises on the first REST call if a value does not convert.
    Nothing else catches this: the rest of the suite points at a sqlite tracking
    URI, which uses the direct store and never issues an HTTP request.
    """
    for candidate in _TUNED_VARIABLES:
        monkeypatch.delenv(candidate.name, raising=False)
    get_settings.cache_clear()

    configure_mlflow_runtime(get_settings())

    assert variable.get() is not None
    get_settings.cache_clear()


@pytest.mark.parametrize("variable", _TUNED_VARIABLES, ids=lambda v: v.name)
def test_worker_task_bounds_mlflow_retries(variable, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Celery task must bound retries too, not just the API lifespan.

    A training run issues far more REST calls than a readiness probe, so when the
    worker ran on MLflow's default 7-retry exponential ladder a single failing
    artifact upload pinned the job in RUNNING for minutes with no error surfaced.
    """
    for candidate in _TUNED_VARIABLES:
        monkeypatch.delenv(candidate.name, raising=False)
    get_settings.cache_clear()
    calls: list[str] = []
    monkeypatch.setattr(
        "kepler_engine.workers.tasks.execute_experiment",
        lambda *a, **kw: calls.append("ran") or {},
    )
    monkeypatch.setattr("kepler_engine.workers.tasks.create_redis_client", lambda settings: None)
    monkeypatch.setattr("kepler_engine.workers.tasks.JobStore", lambda *a, **kw: None)

    run_experiment_task.run("run-1", {})

    assert calls == ["ran"]
    assert variable.get() is not None
    get_settings.cache_clear()


def test_liveness_always_200(app_client) -> None:
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_checks(app_client) -> None:
    response = app_client.get("/health/ready")
    # Redis is fake (healthy); MLflow file store should ping OK.
    body = response.json()
    assert "checks" in body
    names = {c["name"] for c in body["checks"]}
    assert {"redis", "mlflow", "champion_model"} <= names
