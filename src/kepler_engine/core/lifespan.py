"""Application lifespan: Redis, MLflow URI, model cache warmup."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import mlflow
from fastapi import FastAPI

from kepler_engine.core.config import get_settings
from kepler_engine.core.logging import get_logger
from kepler_engine.services.inference_service import InferenceService
from kepler_engine.services.job_store import JobStore, create_redis_client
from kepler_engine.services.mlflow_client import MLflowService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    redis_client = create_redis_client(settings)
    job_store = JobStore(redis_client, ttl_seconds=settings.job_ttl_seconds)
    mlflow_service = MLflowService(settings)
    inference = InferenceService(settings=settings, mlflow_service=mlflow_service)

    app.state.settings = settings
    app.state.redis = redis_client
    app.state.job_store = job_store
    app.state.mlflow_service = mlflow_service
    app.state.inference_service = inference

    inference.warm()
    logger.info("app.startup", env=settings.env, mlflow=settings.mlflow_tracking_uri)

    try:
        yield
    finally:
        try:
            redis_client.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info("app.shutdown")
