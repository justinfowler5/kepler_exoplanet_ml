"""Pluggable KOI data sources: local CSV, S3/MinIO, and NASA Exoplanet Archive TAP.

Every source returns a DataFrame with lowercased column names that is guaranteed
to carry the full feature allowlist plus the target column, so the trainer can
treat "where the data came from" as a configuration detail.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import boto3
import httpx
import pandas as pd
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from kepler_engine.core.config import (
    PROJECT_ROOT,
    DataSourceType,
    Settings,
    get_settings,
)
from kepler_engine.core.exceptions import DataIngestionError
from kepler_engine.core.logging import get_logger
from kepler_engine.ml.features import FEATURE_COLUMNS, TARGET_COLUMN

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)
_REQUIRED_SET: frozenset[str] = frozenset(REQUIRED_COLUMNS)

_S3_ERROR_HINTS: dict[str, str] = {
    "NoSuchBucket": "bucket does not exist",
    "NoSuchKey": "object does not exist",
    "404": "object does not exist",
    "AccessDenied": "access denied; check credentials and bucket policy",
    "403": "access denied; check credentials and bucket policy",
    "InvalidAccessKeyId": "unknown access key id",
    "SignatureDoesNotMatch": "secret access key does not match the access key id",
}


class KeplerDataSource(Protocol):
    """A source of the KOI table. Implementations must be safe to reuse."""

    def load(self) -> pd.DataFrame: ...


def _read_csv(source: Path | io.BytesIO, *, origin: str) -> pd.DataFrame:
    """Parse a KOI CSV, then normalize and validate its schema.

    ``comment="#"`` matters: the KOI table downloaded from the Exoplanet Archive
    web UI begins with a metadata block of ``#``-prefixed lines, which would
    otherwise be parsed as the header row. Pandas tracks quote state, so a ``#``
    inside a quoted field is left intact.
    """
    try:
        df = pd.read_csv(source, comment="#", skipinitialspace=True, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DataIngestionError(f"{origin} contains no CSV data") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        raise DataIngestionError(f"Failed to parse CSV from {origin}: {exc}") from exc

    df.columns = [_normalize_column(column) for column in df.columns]
    return _validate_schema(df, origin=origin)


def _normalize_column(name: object) -> str:
    """Canonicalize a header cell.

    TAP exports and archive downloads do not guarantee header casing, and
    lowercasing here lets the rest of the codebase rely on the names in
    ``FEATURE_COLUMNS``. Dropping the byte-order mark is belt-and-braces for
    Windows-authored CSVs whose BOM pandas did not absorb into the first field.
    """
    return str(name).replace("\ufeff", "").strip().lower()


def _validate_schema(df: pd.DataFrame, *, origin: str) -> pd.DataFrame:
    """Reject frames the trainer cannot consume, with an actionable message."""
    duplicated = sorted({c for c in df.columns[df.columns.duplicated()] if c in _REQUIRED_SET})
    if duplicated:
        # Selecting a duplicated name yields a DataFrame instead of a Series and
        # breaks feature selection several layers downstream.
        raise DataIngestionError(f"{origin} declares duplicate required columns: {duplicated}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        detail = f"{origin} is missing required columns: {missing}"
        if len(missing) == len(REQUIRED_COLUMNS):
            detail += "; none of them are present, so this is probably not a KOI table"
        raise DataIngestionError(detail)

    if df.empty:
        raise DataIngestionError(f"{origin} has a valid header but no data rows")

    return df


def _resolve_local_path(path: str | Path) -> Path:
    """Resolve *path*, falling back to the repo root for relative paths.

    Settings carry repo-relative defaults such as
    ``data/samples/kepler_koi_sample.csv``, which a Celery worker or script
    launched from a subdirectory would otherwise fail to find. The repo root is
    only a fallback because it is meaningless for an installed, non-editable
    package — the Docker case, where the path comes from the environment.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    fallback = PROJECT_ROOT / candidate
    return fallback if fallback.is_file() else candidate


class LocalCSVDataSource:
    """Reads the KOI table from the local filesystem."""

    def __init__(self, path: str | Path) -> None:
        self.path = _resolve_local_path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.is_file():
            raise DataIngestionError(f"Local CSV not found: {self.path}")
        df = _read_csv(self.path, origin=str(self.path))
        logger.info(
            "ingestion.loaded",
            source=DataSourceType.LOCAL_CSV.value,
            path=str(self.path),
            rows=len(df),
            columns=len(df.columns),
        )
        return df


class S3DataSource:
    """Reads the KOI table from S3, or any S3-compatible endpoint such as MinIO.

    Credentials are optional: omitting them lets botocore use its normal
    resolution chain, which is how the EKS pods authenticate via an IRSA
    service-account role instead of static keys.
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        *,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
    ) -> None:
        self.bucket = bucket
        self.key = key
        self.region = region
        self.endpoint_url = endpoint_url
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._client = None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def _get_client(self):
        """Build the client on first use.

        ``ExperimentTrainer`` resolves its data source during construction, so an
        eagerly built client would turn a missing credential into a failure at
        import or worker startup rather than a failed load with a clear message.
        """
        if self._client is None:
            config = BotoConfig(
                region_name=self.region,
                connect_timeout=self._connect_timeout,
                read_timeout=self._read_timeout,
                retries={"max_attempts": 3, "mode": "standard"},
            )
            try:
                self._client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self._access_key_id,
                    aws_secret_access_key=self._secret_access_key,
                    config=config,
                )
            except (BotoCoreError, ValueError) as exc:
                raise DataIngestionError(f"Failed to create an S3 client: {exc}") from exc
        return self._client

    def load(self) -> pd.DataFrame:
        client = self._get_client()
        try:
            response = client.get_object(Bucket=self.bucket, Key=self.key)
            body = response["Body"].read()
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", "Unknown"))
            hint = _S3_ERROR_HINTS.get(code, str(exc))
            raise DataIngestionError(f"Cannot read {self.uri}: {hint} ({code})") from exc
        except BotoCoreError as exc:
            raise DataIngestionError(f"Cannot reach S3 for {self.uri}: {exc}") from exc

        df = _read_csv(io.BytesIO(body), origin=self.uri)
        logger.info(
            "ingestion.loaded",
            source=DataSourceType.S3.value,
            uri=self.uri,
            bytes=len(body),
            rows=len(df),
            columns=len(df.columns),
        )
        return df


class NasaArchiveDataSource:
    """Fetches the KOI cumulative table live from the NASA Exoplanet Archive TAP service."""

    def __init__(self, url: str, *, timeout: float = 120.0) -> None:
        self.url = url
        self.timeout = timeout

    def load(self) -> pd.DataFrame:
        try:
            response = httpx.get(self.url, timeout=self.timeout, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DataIngestionError(
                f"NASA archive returned HTTP {exc.response.status_code} for {self.url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise DataIngestionError(f"Cannot reach the NASA archive at {self.url}: {exc}") from exc

        content = response.content
        _reject_tap_error(content, url=self.url)

        df = _read_csv(io.BytesIO(content), origin="the NASA archive TAP response")
        logger.info(
            "ingestion.loaded",
            source=DataSourceType.NASA_ARCHIVE.value,
            url=self.url,
            bytes=len(content),
            rows=len(df),
            columns=len(df.columns),
        )
        return df


def _reject_tap_error(content: bytes, *, url: str) -> None:
    """Raise if a 200 response body is a TAP error rather than CSV.

    The TAP sync endpoint reports a malformed ADQL query with HTTP 200 and a
    plain-text or VOTable body. That body parses into a single-column frame, so
    without this check the real message is lost behind a generic complaint about
    missing columns.
    """
    head = content.lstrip()[:512]
    if not head:
        raise DataIngestionError(f"NASA archive returned an empty response for {url}")
    if head.upper().startswith(b"ERROR") or head.startswith(b"<"):
        detail = head.decode("utf-8", errors="replace").strip().split("\n", 1)[0]
        raise DataIngestionError(f"NASA archive rejected the query at {url}: {detail}")


def _coerce_source_type(value: DataSourceType | str) -> DataSourceType:
    if isinstance(value, DataSourceType):
        return value
    try:
        return DataSourceType(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(t.value for t in DataSourceType)
        raise DataIngestionError(
            f"Unknown data source '{value}'; expected one of: {allowed}"
        ) from exc


def get_data_source(
    settings: Settings | None = None,
    override: DataSourceType | str | None = None,
) -> KeplerDataSource:
    """Select the configured data source, or *override* for a single experiment."""
    settings = settings or get_settings()
    source_type = settings.data_source if override is None else _coerce_source_type(override)

    if source_type is DataSourceType.LOCAL_CSV:
        return LocalCSVDataSource(settings.local_csv_path)
    if source_type is DataSourceType.S3:
        return S3DataSource(
            settings.s3_bucket,
            settings.s3_key,
            region=settings.aws_region,
            endpoint_url=settings.s3_endpoint_url,
            access_key_id=settings.aws_access_key_id,
            secret_access_key=settings.aws_secret_access_key,
        )
    if source_type is DataSourceType.NASA_ARCHIVE:
        return NasaArchiveDataSource(settings.nasa_tap_url)
    raise DataIngestionError(f"Unsupported data source: {source_type}")
