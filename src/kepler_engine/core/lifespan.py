"""Application lifespan: Redis, MLflow URI, model cache warmup."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from kepler_engine.core.config import get_settings
from kepler_engine.core.logging import get_logger
from kepler_engine.core.mlflow_runtime import configure_mlflow_runtime
from kepler_engine.services.inference_service import InferenceService
from kepler_engine.services.job_store import JobStore, create_redis_client
from kepler_engine.services.mlflow_client import MLflowService

logger = get_logger(__name__)


def _warm_in_background(inference: InferenceService) -> threading.Thread:
    """Warm the champion-model cache off the startup path.

    Blocking here would let the liveness probe kill the pod before startup finishes
    whenever the registry is slow or cold, so the load runs in a daemon thread and
    /health/ready reports the model's availability in the meantime.
    """

    def _warm() -> None:
        try:
            logger.info("app.model_warm", warmed=inference.warm())
        except Exception as exc:  # noqa: BLE001
            logger.warning("app.warm_failed", error=str(exc))

    thread = threading.Thread(target=_warm, name="model-warm", daemon=True)
    thread.start()
    return thread


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_mlflow_runtime(settings)

    redis_client = create_redis_client(settings)
    job_store = JobStore(redis_client, ttl_seconds=settings.job_ttl_seconds)
    mlflow_service = MLflowService(settings)
    inference = InferenceService(settings=settings, mlflow_service=mlflow_service)

    app.state.settings = settings
    app.state.redis = redis_client
    app.state.job_store = job_store
    app.state.mlflow_service = mlflow_service
    app.state.inference_service = inference

    _warm_in_background(inference)
    logger.info("app.startup", env=settings.env, mlflow=settings.mlflow_tracking_uri)

    try:
        yield
    finally:
        try:
            redis_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("app.redis_close_failed", error=str(exc))
        logger.info("app.shutdown")
