"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.services.inference_service import InferenceService
from kepler_engine.services.job_store import JobStore
from kepler_engine.services.mlflow_client import MLflowService


def settings_dep() -> Settings:
    return get_settings()


def job_store_dep(request: Request) -> JobStore:
    return request.app.state.job_store


def inference_dep(request: Request) -> InferenceService:
    return request.app.state.inference_service


def mlflow_dep(request: Request) -> MLflowService:
    return request.app.state.mlflow_service


def redis_dep(request: Request) -> Any:
    return request.app.state.redis


SettingsDep = Annotated[Settings, Depends(settings_dep)]
JobStoreDep = Annotated[JobStore, Depends(job_store_dep)]
InferenceDep = Annotated[InferenceService, Depends(inference_dep)]
MLflowDep = Annotated[MLflowService, Depends(mlflow_dep)]
