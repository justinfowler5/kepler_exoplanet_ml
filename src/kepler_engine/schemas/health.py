"""Health / readiness schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class DependencyCheck(BaseModel):
    name: str
    healthy: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    checks: list[DependencyCheck] = Field(default_factory=list)
