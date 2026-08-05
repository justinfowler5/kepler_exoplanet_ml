"""Pluggable KOI data sources: S3, local CSV, NASA TAP."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import boto3
import httpx
import pandas as pd

from kepler_engine.core.config import DataSourceType, Settings, get_settings
from kepler_engine.core.exceptions import DataIngestionError
from kepler_engine.core.logging import get_logger
from kepler_engine.ml.features import FEATURE_COLUMNS, TARGET_COLUMN

logger = get_logger(__name__)

REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


class KeplerDataSource(Protocol):
    def load(self) -> pd.DataFrame: ...


def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataIngestionError(f"Dataset missing required columns: {missing}")
    return df


class LocalCSVDataSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataIngestionError(f"Local CSV not found: {self.path}")
        try:
            df = pd.read_csv(self.path)
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"Failed to read local CSV: {exc}") from exc
        logger.info("ingestion.local_csv", path=str(self.path), rows=len(df))
        return _validate_schema(df)


class S3DataSource:
    def __init__(
        self,
        bucket: str,
        key: str,
        *,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.key = key
        session_kwargs: dict = {"region_name": region}
        client_kwargs: dict = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if access_key_id and secret_access_key:
            session_kwargs["aws_access_key_id"] = access_key_id
            session_kwargs["aws_secret_access_key"] = secret_access_key
        session = boto3.session.Session(**session_kwargs)
        self._client = session.client("s3", **client_kwargs)

    def load(self) -> pd.DataFrame:
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=self.key)
            body = obj["Body"].read()
            df = pd.read_csv(io.BytesIO(body))
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(
                f"Failed to load s3://{self.bucket}/{self.key}: {exc}"
            ) from exc
        logger.info("ingestion.s3", bucket=self.bucket, key=self.key, rows=len(df))
        return _validate_schema(df)


class NasaArchiveDataSource:
    def __init__(self, url: str, *, timeout: float = 120.0) -> None:
        self.url = url
        self.timeout = timeout

    def load(self) -> pd.DataFrame:
        try:
            response = httpx.get(self.url, timeout=self.timeout, follow_redirects=True)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content))
        except Exception as exc:  # noqa: BLE001
            raise DataIngestionError(f"Failed to fetch NASA TAP data: {exc}") from exc
        logger.info("ingestion.nasa_archive", rows=len(df))
        return _validate_schema(df)


def get_data_source(settings: Settings | None = None) -> KeplerDataSource:
    settings = settings or get_settings()
    if settings.data_source is DataSourceType.LOCAL_CSV:
        return LocalCSVDataSource(settings.local_csv_path)
    if settings.data_source is DataSourceType.S3:
        return S3DataSource(
            settings.s3_bucket,
            settings.s3_key,
            region=settings.aws_region,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.aws_access_key_id,
            secret_access_key=settings.aws_secret_access_key,
        )
    if settings.data_source is DataSourceType.NASA_ARCHIVE:
        return NasaArchiveDataSource(settings.nasa_tap_url)
    raise DataIngestionError(f"Unknown data source: {settings.data_source}")
