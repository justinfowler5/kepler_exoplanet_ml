"""Celery application for async training jobs."""

from __future__ import annotations

from celery import Celery
from celery.signals import worker_ready

from kepler_engine.core.config import get_settings
from kepler_engine.core.logging import get_logger
from kepler_engine.core.metrics import start_worker_metrics_server

settings = get_settings()
logger = get_logger(__name__)

celery_app = Celery(
    "kepler_engine",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["kepler_engine.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    enable_utc=True,
    timezone="UTC",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=60 * 30,  # 30 minutes hard limit
    task_soft_time_limit=60 * 25,
    broker_connection_retry_on_startup=True,
)


@worker_ready.connect
def _expose_worker_metrics(**_kwargs: object) -> None:
    port = settings.worker_metrics_port
    if port <= 0:
        return
    start_worker_metrics_server(port)
    logger.info("worker.metrics_listening", port=port)


# Eager mode for tests is toggled via CELERY_TASK_ALWAYS_EAGER env / conf in conftest.
