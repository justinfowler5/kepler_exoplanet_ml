"""Application settings via pydantic-settings (KEPLER_ prefix)."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSourceType(str, Enum):
    S3 = "s3"
    LOCAL_CSV = "local_csv"
    NASA_ARCHIVE = "nasa_archive"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KEPLER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"

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
    job_ttl_seconds: int = 86_400

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = "kepler-koi"
    registered_model_name: str = "kepler-koi-classifier"
    model_alias: str = "champion"
    model_cache_ttl_seconds: int = 60

    # Training / promotion
    promote_metric: str = "f1"
    promote_threshold: float = 0.70
    default_test_size: float = 0.2
    default_cv_folds: int = 5
    random_state: int = 42

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
