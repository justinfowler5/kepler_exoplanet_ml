"""RFC 9457 problem+json error handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from kepler_engine.core.exceptions import (
    DataIngestionError,
    ExperimentNotFoundError,
    KeplerEngineError,
    LeakageViolationError,
    ModelNotFoundError,
)


def _problem(
    *,
    status: int,
    title: str,
    detail: str,
    type_: str = "about:blank",
    instance: str | None = None,
    extras: dict | None = None,
) -> JSONResponse:
    body: dict = {
        "type": type_,
        "title": title,
        "status": status,
        "detail": detail,
    }
    if instance:
        body["instance"] = instance
    if extras:
        body.update(extras)
    return JSONResponse(status_code=status, content=body, media_type="application/problem+json")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ModelNotFoundError)
    async def model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return _problem(
            status=404,
            title="Model Not Found",
            detail=str(exc),
            type_="https://kepler.engine/errors/model-not-found",
            instance=str(request.url),
        )

    @app.exception_handler(ExperimentNotFoundError)
    async def experiment_not_found(
        request: Request, exc: ExperimentNotFoundError
    ) -> JSONResponse:
        return _problem(
            status=404,
            title="Experiment Not Found",
            detail=str(exc),
            type_="https://kepler.engine/errors/experiment-not-found",
            instance=str(request.url),
        )

    @app.exception_handler(LeakageViolationError)
    async def leakage(request: Request, exc: LeakageViolationError) -> JSONResponse:
        return _problem(
            status=422,
            title="Leakage Violation",
            detail=str(exc),
            type_="https://kepler.engine/errors/leakage-violation",
            instance=str(request.url),
        )

    @app.exception_handler(DataIngestionError)
    async def ingestion(request: Request, exc: DataIngestionError) -> JSONResponse:
        return _problem(
            status=502,
            title="Data Ingestion Error",
            detail=str(exc),
            type_="https://kepler.engine/errors/data-ingestion",
            instance=str(request.url),
        )

    @app.exception_handler(KeplerEngineError)
    async def domain(request: Request, exc: KeplerEngineError) -> JSONResponse:
        return _problem(
            status=400,
            title="Kepler Engine Error",
            detail=str(exc),
            instance=str(request.url),
        )

    @app.exception_handler(RequestValidationError)
    async def validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _problem(
            status=422,
            title="Validation Error",
            detail="Request validation failed",
            type_="https://kepler.engine/errors/validation",
            instance=str(request.url),
            extras={"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exc(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem(
            status=exc.status_code,
            title="HTTP Error",
            detail=str(exc.detail),
            instance=str(request.url),
        )
