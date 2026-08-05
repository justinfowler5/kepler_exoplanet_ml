"""Health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from kepler_engine.schemas.health import DependencyCheck, HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def liveness() -> HealthResponse:
    """Liveness probe — always 200 while the process is alive."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Readiness probe — Redis, MLflow, and champion model loadability."""
    checks: list[DependencyCheck] = []

    # Redis
    try:
        request.app.state.redis.ping()
        checks.append(DependencyCheck(name="redis", healthy=True))
    except Exception as exc:  # noqa: BLE001
        checks.append(DependencyCheck(name="redis", healthy=False, detail=str(exc)))

    # MLflow
    try:
        healthy = request.app.state.mlflow_service.ping()
        checks.append(
            DependencyCheck(
                name="mlflow",
                healthy=healthy,
                detail=None if healthy else "unreachable",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(DependencyCheck(name="mlflow", healthy=False, detail=str(exc)))

    # Champion model — optional at boot (no model yet is not a hard fail for first deploy)
    try:
        loadable = request.app.state.inference_service.is_loadable()
        checks.append(
            DependencyCheck(
                name="champion_model",
                healthy=True,
                detail="loaded" if loadable else "no champion yet",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(DependencyCheck(name="champion_model", healthy=False, detail=str(exc)))

    # Ready if redis + mlflow are healthy; champion may be absent before first train.
    critical_ok = all(c.healthy for c in checks if c.name in {"redis", "mlflow"})
    status = "ready" if critical_ok else "not_ready"
    if not critical_ok:
        response.status_code = 503
    return ReadinessResponse(status=status, checks=checks)
