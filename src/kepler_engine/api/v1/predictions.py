"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from kepler_engine.api.deps import InferenceDep
from kepler_engine.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest, inference: InferenceDep) -> PredictionResponse:
    records = [r.model_dump() for r in body.records]
    results = inference.predict(records)
    version = inference.model_version
    return PredictionResponse(
        predictions=[PredictionResult(**r) for r in results],
        model_version=version,
    )
