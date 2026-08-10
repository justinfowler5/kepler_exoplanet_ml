"""Experiment enqueue (API) and execute (worker) orchestration."""

from __future__ import annotations

from typing import Any

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.core.logging import get_logger
from kepler_engine.core.metrics import observe_training, track_model_operation
from kepler_engine.ml.labels import LabelStrategy
from kepler_engine.ml.models import ModelType
from kepler_engine.ml.trainer import ExperimentTrainer
from kepler_engine.services.job_store import JobStore

logger = get_logger(__name__)


def enqueue_experiment(
    job_store: JobStore,
    request: dict[str, Any],
) -> str:
    """Create a PENDING job and enqueue the Celery task. Returns run_id."""
    run_id = job_store.create(payload=request)
    # Late import keeps Celery out of the API import graph when eager/testing.
    from kepler_engine.workers.tasks import run_experiment_task

    run_experiment_task.delay(run_id, request)
    logger.info("experiment.enqueued", run_id=run_id, model_type=request.get("model_type"))
    return run_id


def execute_experiment(
    run_id: str,
    request: dict[str, Any],
    *,
    job_store: JobStore,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Worker-side: run training and update job status."""
    settings = settings or get_settings()
    model_type = str(request.get("model_type", ModelType.RANDOM_FOREST.value))
    job_store.mark_running(run_id, progress=0.1)

    op: dict[str, Any] = {}
    try:
        with track_model_operation("training") as op:
            trainer = ExperimentTrainer(settings=settings)
            result = trainer.run(
                model_type=ModelType(model_type),
                hyperparams=request.get("hyperparams"),
                test_size=request.get("test_size"),
                cv_folds=request.get("cv_folds"),
                label_strategy=LabelStrategy(
                    request.get("label_strategy", LabelStrategy.BINARY.value)
                ),
                promote=request.get("promote", True),
            )
    finally:
        if op:
            observe_training(
                model_type,
                str(op.get("status", "error")),
                float(op.get("duration_seconds", 0.0)),
            )

    job_store.mark_success(
        run_id,
        metrics=result.get("metrics", {}),
        model_version=result.get("model_version"),
        mlflow_run_id=result.get("run_id"),
    )
    # Attach extra fields for callers that read the full job record.
    job_store.update(
        run_id,
        labels=result.get("labels"),
        promoted=result.get("promoted"),
        model_uri=result.get("model_uri"),
        progress=1.0,
    )
    return result