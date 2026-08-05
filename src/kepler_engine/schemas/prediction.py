"""Prediction request/response schemas with physical bounds."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransitMetrics(BaseModel):
    """One record of allowlisted KOI transit / stellar measurements."""

    koi_period: float = Field(..., gt=0, description="Orbital period (days)")
    koi_time0bk: float = Field(..., description="Transit epoch BJD-2454833")
    koi_impact: float | None = Field(default=None, ge=0, le=2)
    koi_duration: float = Field(..., gt=0, description="Transit duration (hours)")
    koi_depth: float | None = Field(default=None, ge=0, description="Transit depth (ppm)")
    koi_prad: float | None = Field(default=None, gt=0, description="Planet radius (Earth)")
    koi_teq: float | None = Field(default=None, gt=0, description="Equilibrium temp (K)")
    koi_insol: float | None = Field(default=None, ge=0, description="Insolation (Earth flux)")
    koi_model_snr: float | None = Field(default=None, ge=0)
    koi_tce_plnt_num: int = Field(..., ge=1, le=10)
    koi_steff: float | None = Field(default=None, gt=0, description="Stellar Teff (K)")
    koi_slogg: float | None = Field(default=None, gt=0, description="log g")
    koi_srad: float | None = Field(default=None, gt=0, description="Stellar radius (solar)")
    koi_kepmag: float = Field(..., gt=0, le=25, description="Kepler magnitude")


class PredictionRequest(BaseModel):
    records: list[TransitMetrics] = Field(..., min_length=1, max_length=500)


class PredictionResult(BaseModel):
    label: str
    probability: float | None = None
    model_version: str | None = None
    run_id: str | None = None


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]
    model_version: str | None = None
