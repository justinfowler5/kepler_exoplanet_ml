"""Celery application for async training jobs."""

from __future__ import annotations

from celery import Celery

from kepler_engine.core.config import get_settings

settings = get_settings()

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

# Eager mode for tests is toggled via CELERY_TASK_ALWAYS_EAGER env / conf in conftest.
