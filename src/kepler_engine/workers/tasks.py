"""Celery tasks."""

from __future__ import annotations

from typing import Any

from kepler_engine.core.config import get_settings
from kepler_engine.core.logging import configure_logging, get_logger
from kepler_engine.core.mlflow_runtime import configure_mlflow_runtime
from kepler_engine.services.experiment_service import execute_experiment
from kepler_engine.services.job_store import JobStore, create_redis_client
from kepler_engine.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="kepler_engine.run_experiment")
def run_experiment_task(self, run_id: str, request: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.env != "local")
    configure_mlflow_runtime(settings)
    redis_client = create_redis_client(settings)
    job_store = JobStore(redis_client, ttl_seconds=settings.job_ttl_seconds)

    try:
        logger.info("task.start", run_id=run_id, celery_id=self.request.id)
        result = execute_experiment(run_id, request, job_store=job_store, settings=settings)
        logger.info("task.success", run_id=run_id, mlflow_run_id=result.get("run_id"))
        return {
            "run_id": run_id,
            "mlflow_run_id": result.get("run_id"),
            "metrics": result.get("metrics"),
            "model_version": result.get("model_version"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("task.failure", run_id=run_id, error=str(exc))
        try:
            job_store.mark_failure(run_id, error=str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("task.failure_status_update_failed", run_id=run_id)
        raise
