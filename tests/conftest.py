"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import fakeredis
import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Eager Celery before app import so tasks run inline.
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("KEPLER_ENV", "local")
os.environ.setdefault("KEPLER_DATA_SOURCE", "local_csv")
os.environ.setdefault(
    "KEPLER_LOCAL_CSV_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "samples" / "kepler_koi_sample.csv"),
)
os.environ.setdefault("KEPLER_PROMOTE_THRESHOLD", "0.0")

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "samples" / "kepler_koi_sample.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def feature_only_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    from kepler_engine.ml.features import FEATURE_COLUMNS, TARGET_COLUMN

    return sample_df[[*FEATURE_COLUMNS, TARGET_COLUMN]].copy()


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def tmp_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # MLflow 3.15+ rejects the legacy file store unless opted in; use sqlite in tests.
    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    from kepler_engine.core.config import get_settings

    get_settings.cache_clear()
    return tmp_path


@pytest.fixture
def app_client(fake_redis, tmp_mlflow, monkeypatch: pytest.MonkeyPatch):
    from kepler_engine.core.config import get_settings
    from kepler_engine.main import create_app
    from kepler_engine.services.inference_service import InferenceService
    from kepler_engine.services.job_store import JobStore
    from kepler_engine.services.mlflow_client import MLflowService
    from kepler_engine.workers import celery_app as celery_module

    get_settings.cache_clear()
    celery_module.celery_app.conf.task_always_eager = True
    celery_module.celery_app.conf.task_eager_propagates = True

    # Avoid real Redis during lifespan by pointing KEPLER_REDIS_URL at a dummy;
    # we overwrite app.state after startup.
    monkeypatch.setenv("KEPLER_REDIS_URL", "redis://localhost:6379/15")
    get_settings.cache_clear()

    app = create_app()
    job_store = JobStore(fake_redis, ttl_seconds=3600)
    mlflow_service = MLflowService(get_settings())
    inference = InferenceService(settings=get_settings(), mlflow_service=mlflow_service)

    # Patch lifespan redis creation
    monkeypatch.setattr(
        "kepler_engine.core.lifespan.create_redis_client",
        lambda settings=None: fake_redis,
    )

    with TestClient(app) as client:
        app.state.redis = fake_redis
        app.state.job_store = job_store
        app.state.mlflow_service = mlflow_service
        app.state.inference_service = inference
        app.state.settings = get_settings()
        yield client

    get_settings.cache_clear()
