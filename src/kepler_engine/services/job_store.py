"""Redis-backed job lifecycle store."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

import redis

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.core.exceptions import ExperimentNotFoundError


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    """Redis hash per job: status, progress, metrics, error, timestamps."""

    def __init__(
        self,
        redis_client: redis.Redis,
        *,
        ttl_seconds: int = 86_400,
        key_prefix: str = "kepler:job:",
    ) -> None:
        self._redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    def _key(self, run_id: str) -> str:
        return f"{self.key_prefix}{run_id}"

    def create(self, *, payload: dict[str, Any] | None = None) -> str:
        run_id = str(uuid4())
        now = _utcnow()
        record = {
            "run_id": run_id,
            "status": JobStatus.PENDING.value,
            "progress": 0.0,
            "metrics": None,
            "error": None,
            "model_version": None,
            "mlflow_run_id": None,
            "payload": payload or {},
            "created_at": now,
            "updated_at": now,
        }
        self._write(run_id, record)
        return run_id

    def get(self, run_id: str) -> dict[str, Any]:
        raw = self._redis.get(self._key(run_id))
        if raw is None:
            raise ExperimentNotFoundError(f"Unknown run_id: {run_id}")
        return json.loads(raw)

    def update(self, run_id: str, **fields: Any) -> dict[str, Any]:
        record = self.get(run_id)
        record.update(fields)
        record["updated_at"] = _utcnow()
        self._write(run_id, record)
        return record

    def mark_running(self, run_id: str, *, progress: float = 0.1) -> dict[str, Any]:
        return self.update(run_id, status=JobStatus.RUNNING.value, progress=progress)

    def mark_success(
        self,
        run_id: str,
        *,
        metrics: dict[str, Any],
        model_version: str | None = None,
        mlflow_run_id: str | None = None,
    ) -> dict[str, Any]:
        return self.update(
            run_id,
            status=JobStatus.SUCCESS.value,
            progress=1.0,
            metrics=metrics,
            model_version=model_version,
            mlflow_run_id=mlflow_run_id,
            error=None,
        )

    def mark_failure(self, run_id: str, error: str) -> dict[str, Any]:
        return self.update(
            run_id,
            status=JobStatus.FAILURE.value,
            error=error,
        )

    def _write(self, run_id: str, record: dict[str, Any]) -> None:
        self._redis.set(
            self._key(run_id),
            json.dumps(record, default=str),
            ex=self.ttl_seconds,
        )


def create_redis_client(settings: Settings | None = None) -> redis.Redis:
    settings = settings or get_settings()
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=settings.redis_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )
