"""Experiment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from kepler_engine.api.deps import JobStoreDep, MLflowDep
from kepler_engine.schemas.experiment import (
    ExperimentAccepted,
    ExperimentListItem,
    ExperimentRunRequest,
    ExperimentStatus,
)
from kepler_engine.services.experiment_service import enqueue_experiment

router = APIRouter(prefix="/experiment", tags=["experiments"])


@router.post(
    "/run",
    response_model=ExperimentAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_experiment(body: ExperimentRunRequest, job_store: JobStoreDep) -> ExperimentAccepted:
    run_id = enqueue_experiment(job_store, body.model_dump(mode="json"))
    return ExperimentAccepted(run_id=run_id, status="PENDING")


@router.get("/{run_id}", response_model=ExperimentStatus)
def get_experiment(run_id: str, job_store: JobStoreDep) -> ExperimentStatus:
    record = job_store.get(run_id)
    return ExperimentStatus(
        run_id=record["run_id"],
        status=record["status"],
        progress=float(record.get("progress") or 0),
        metrics=record.get("metrics"),
        error=record.get("error"),
        model_version=record.get("model_version"),
        mlflow_run_id=record.get("mlflow_run_id"),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
        promoted=record.get("promoted"),
    )


@router.get("", response_model=list[ExperimentListItem])
def list_experiments(mlflow_service: MLflowDep, limit: int = 20) -> list[ExperimentListItem]:
    runs = mlflow_service.list_recent_runs(max_results=min(limit, 100))
    return [
        ExperimentListItem(
            run_id=r["run_id"],
            status=r.get("status"),
            metrics=r.get("metrics") or {},
            params=r.get("params") or {},
            start_time=r.get("start_time"),
        )
        for r in runs
    ]
