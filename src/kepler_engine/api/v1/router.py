"""Aggregate /api/v1 routers."""

from fastapi import APIRouter

from kepler_engine.api.v1 import experiments, predictions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(experiments.router)
api_router.include_router(predictions.router)
