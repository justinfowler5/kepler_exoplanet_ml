"""Application settings via pydantic-settings (KEPLER_ prefix)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# <root>/src/kepler_engine/core/config.py -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DataSourceType(StrEnum):
    S3 = "s3"
    LOCAL_CSV = "local_csv"
    NASA_ARCHIVE = "nasa_archive"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KEPLER_",
        # The repo-root fallback keeps `celery -A ...` and scripts working when they
        # are launched from a subdirectory. Missing files are ignored, so the
        # installed-package case (Docker, where config comes from the environment)
        # is unaffected.
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["local", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Data
    data_source: DataSourceType = DataSourceType.LOCAL_CSV
    local_csv_path: str = "data/samples/kepler_koi_sample.csv"
    s3_bucket: str = "kepler-koi-data"
    s3_key: str = "cumulative.csv"
    s3_endpoint_url: str | None = None
    nasa_tap_url: str = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        "?query=select+*+from+cumulative&format=csv"
    )

    # AWS (also readable without prefix via Field aliases when needed)
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(
        default=None, validation_alias="AWS_SECRET_ACCESS_KEY"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    # Without an explicit connect timeout a black-holed Redis hangs the readiness
    # probe for the OS TCP timeout. Budget is doubled in practice: hostnames that
    # resolve to both IPv6 and IPv4 are attempted in turn.
    redis_connect_timeout_seconds: float = Field(default=1.0, gt=0)
    redis_socket_timeout_seconds: float = Field(default=2.0, gt=0)
    job_ttl_seconds: int = Field(default=86_400, gt=0)

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = "kepler-koi"
    # MLflow defaults to 7 retries with exponential backoff, which makes a single call
    # against an unreachable tracking server block for minutes. These bound the API's
    # calls so /health/ready answers well inside its 5s probe timeout; the Celery
    # worker keeps MLflow's patient defaults, which suit a long training job.
    mlflow_http_max_retries: int = Field(default=1, ge=0)
    mlflow_http_timeout_seconds: int = Field(default=5, gt=0)
    # MLflow parses MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR with int(), not float() —
    # unlike the neighbouring BACKOFF_JITTER — so a fractional value here makes
    # every REST call raise ValueError before it is sent. 0 means retry without
    # sleeping, which is what the bounded probe path wants anyway.
    mlflow_http_backoff_factor: int = Field(default=0, ge=0)
    registered_model_name: str = "kepler-koi-classifier"
    model_alias: str = "champion"
    model_cache_ttl_seconds: int = Field(default=60, ge=0)

    # Training / promotion
    promote_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] = "f1"
    promote_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    default_test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    default_cv_folds: int = Field(default=5, ge=2)
    random_state: int = 42

    # API
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65_535)

    # Worker Prometheus scrape endpoint (0 disables the side server)
    worker_metrics_port: int = Field(default=9100, ge=0, le=65_535)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
