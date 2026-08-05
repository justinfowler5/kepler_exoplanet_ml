"""Ingestion source tests."""

from __future__ import annotations

from pathlib import Path

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from kepler_engine.core.exceptions import DataIngestionError
from kepler_engine.services.ingestion import LocalCSVDataSource, S3DataSource

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
