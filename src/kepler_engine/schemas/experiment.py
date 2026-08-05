"""Experiment request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kepler_engine.ml.labels import LabelStrategy
from kepler_engine.ml.models import ModelType


class ExperimentRunRequest(BaseModel):
    model_type: ModelType = ModelType.RANDOM_FOREST
    hyperparams: dict[str, Any] | None = None
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    label_strategy: LabelStrategy = LabelStrategy.BINARY
    promote: bool = True
    data_source: str | None = Field(
        default=None,
        description="Optional override: s3 | local_csv | nasa_archive",
    )


class ExperimentAccepted(BaseModel):
    run_id: str
    status: str = "PENDING"
    message: str = "Experiment enqueued"


class ExperimentMetrics(BaseModel):
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None


class ExperimentStatus(BaseModel):
    run_id: str
    status: str
    progress: float = 0.0
    metrics: ExperimentMetrics | dict[str, Any] | None = None
    error: str | None = None
    model_version: str | None = None
    mlflow_run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    promoted: bool | None = None


class ExperimentListItem(BaseModel):
    run_id: str
    status: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    start_time: int | None = None
