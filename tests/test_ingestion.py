"""Ingestion source tests."""

from __future__ import annotations

from pathlib import Path

import boto3
import httpx
import pandas as pd
import pytest
from moto import mock_aws

from kepler_engine.core.config import DataSourceType, Settings
from kepler_engine.core.exceptions import DataIngestionError
from kepler_engine.services.ingestion import (
    LocalCSVDataSource,
    NasaArchiveDataSource,
    S3DataSource,
    get_data_source,
)

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "samples" / "kepler_koi_sample.csv"


def test_local_csv_loads(sample_df: pd.DataFrame) -> None:
    src = LocalCSVDataSource(SAMPLE_CSV)
    df = src.load()
    assert len(df) == len(sample_df)
    assert "koi_period" in df.columns
    assert "koi_disposition" in df.columns


def test_local_csv_missing_raises() -> None:
    src = LocalCSVDataSource("data/raw/does_not_exist.csv")
    with pytest.raises(DataIngestionError):
        src.load()


@mock_aws
def test_s3_and_local_parity(sample_df: pd.DataFrame) -> None:
    bucket = "kepler-koi-data"
    key = "cumulative.csv"
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=bucket)
    client.put_object(Bucket=bucket, Key=key, Body=SAMPLE_CSV.read_bytes())

    s3_df = S3DataSource(bucket, key, region="us-east-1").load()
    local_df = LocalCSVDataSource(SAMPLE_CSV).load()

    assert list(s3_df.columns) == list(local_df.columns)
    assert len(s3_df) == len(local_df)
    pd.testing.assert_series_equal(
        s3_df["koi_period"].reset_index(drop=True),
        local_df["koi_period"].reset_index(drop=True),
    )


@mock_aws
@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("NoSuchBucket", "bucket does not exist"),
        ("NoSuchKey", "object does not exist"),
        ("AccessDenied", "access denied"),
    ],
)
def test_s3_client_errors_are_translated(error_code: str, expected: str) -> None:
    """A boto ClientError must surface as an actionable DataIngestionError."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="kepler-koi-data")
    if error_code == "NoSuchBucket":
        source = S3DataSource("no-such-bucket", "cumulative.csv", region="us-east-1")
    else:
        source = S3DataSource("kepler-koi-data", "absent.csv", region="us-east-1")

    with pytest.raises(DataIngestionError) as exc:
        source.load()

    message = str(exc.value)
    assert "s3://" in message
    if error_code != "AccessDenied":
        assert expected in message


def _archive_style_csv(path: Path) -> Path:
    """Write the sample table the way the Exoplanet Archive actually serves it.

    Three quirks combine in a real download: a leading block of ``#`` metadata
    lines, a UTF-8 BOM, and upper-cased headers. Any one of them left unhandled
    turns the header row into data or hides every required column.
    """
    frame = pd.read_csv(SAMPLE_CSV)
    frame.columns = [c.upper() for c in frame.columns]
    body = frame.to_csv(index=False)
    preamble = (
        "# This file was produced by the NASA Exoplanet Archive\n"
        "# COLUMN koi_period:  Orbital Period [days]\n"
    )
    path.write_bytes(preamble.encode("utf-8") + b"\xef\xbb\xbf" + body.encode("utf-8"))
    return path


def test_archive_format_quirks_are_normalized(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    path = _archive_style_csv(tmp_path / "archive.csv")

    df = LocalCSVDataSource(path).load()

    assert len(df) == len(sample_df)
    assert "koi_period" in df.columns
    assert "koi_disposition" in df.columns
    assert not any(c.startswith("#") or "\ufeff" in c for c in df.columns)


def test_case_colliding_columns_are_rejected(tmp_path: Path) -> None:
    """Headers that differ only by case collide once normalized, and must be refused.

    Pandas mangles literally identical headers into ``koi_period.1``, so this is
    the reachable route to a duplicate required column: ``KOI_PERIOD`` and
    ``koi_period`` are distinct to the parser but identical after normalization.
    Selecting the collapsed name yields a DataFrame rather than a Series and
    would fail several layers downstream, inside feature selection.
    """
    lines = SAMPLE_CSV.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    first_value = rows[0].split(",")[0]
    path = tmp_path / "case_collision.csv"
    path.write_text(
        "\n".join([f"{header},KOI_PERIOD", *[f"{r},{first_value}" for r in rows]]),
        encoding="utf-8",
    )

    with pytest.raises(DataIngestionError, match="duplicate required columns"):
        LocalCSVDataSource(path).load()


def test_header_only_csv_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "header_only.csv"
    path.write_text(SAMPLE_CSV.read_text(encoding="utf-8").splitlines()[0], encoding="utf-8")

    with pytest.raises(DataIngestionError, match="no data rows"):
        LocalCSVDataSource(path).load()


def test_non_koi_csv_gets_a_specific_hint(tmp_path: Path) -> None:
    path = tmp_path / "unrelated.csv"
    path.write_text("alpha,beta\n1,2\n", encoding="utf-8")

    with pytest.raises(DataIngestionError, match="probably not a KOI table"):
        LocalCSVDataSource(path).load()


@pytest.mark.parametrize("body", [b"ERROR: unable to parse ADQL query", b"<VOTABLE></VOTABLE>"])
def test_tap_error_body_with_http_200_is_rejected(
    body: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TAP reports a bad query with HTTP 200, so the body has to be inspected."""
    url = "https://example.invalid/TAP/sync"

    def _fake_get(*args, **kwargs) -> httpx.Response:
        return httpx.Response(200, content=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", _fake_get)

    with pytest.raises(DataIngestionError, match="rejected the query"):
        NasaArchiveDataSource(url).load()


def test_tap_success_parses(monkeypatch: pytest.MonkeyPatch, sample_df: pd.DataFrame) -> None:
    url = "https://example.invalid/TAP/sync"

    def _fake_get(*args, **kwargs) -> httpx.Response:
        return httpx.Response(
            200, content=SAMPLE_CSV.read_bytes(), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", _fake_get)

    assert len(NasaArchiveDataSource(url).load()) == len(sample_df)


def test_get_data_source_honours_override() -> None:
    settings = Settings(local_csv_path=str(SAMPLE_CSV), data_source=DataSourceType.LOCAL_CSV)

    assert isinstance(get_data_source(settings), LocalCSVDataSource)
    assert isinstance(get_data_source(settings, override="nasa_archive"), NasaArchiveDataSource)
    assert isinstance(get_data_source(settings, override=DataSourceType.S3), S3DataSource)


def test_get_data_source_rejects_unknown_override() -> None:
    settings = Settings(local_csv_path=str(SAMPLE_CSV))

    with pytest.raises(DataIngestionError, match="Unknown data source"):
        get_data_source(settings, override="ftp")
