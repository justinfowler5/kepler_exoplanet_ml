"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from kepler_engine.api.errors import register_exception_handlers
from kepler_engine.api.v1.health import router as health_router
from kepler_engine.api.v1.router import api_router
from kepler_engine.core.config import get_settings
from kepler_engine.core.http_metrics import Http5xxMetricsMiddleware
from kepler_engine.core.lifespan import lifespan
from kepler_engine.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.env != "local")

    app = FastAPI(
        title="Kepler Exoplanet Classification Engine",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "production" else None,
        redoc_url="/redoc" if settings.env != "production" else None,
    )

    register_exception_handlers(app)
    app.add_middleware(Http5xxMetricsMiddleware)
    app.include_router(health_router)
    app.include_router(api_router)

    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health", "/health/ready"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
