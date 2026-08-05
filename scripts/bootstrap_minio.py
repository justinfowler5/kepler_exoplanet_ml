#!/usr/bin/env python3
"""Create the MLflow artifacts bucket in local MinIO."""

from __future__ import annotations

import os
import sys

import boto3
from botocore.exceptions import ClientError


def main() -> int:
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    bucket = os.environ.get("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
    region = os.environ.get("AWS_REGION", "us-east-1")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    try:
        client.head_bucket(Bucket=bucket)
        print(f"Bucket already exists: {bucket}")
    except ClientError:
        params: dict = {"Bucket": bucket}
        if region != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": region}
        client.create_bucket(**params)
        print(f"Created bucket: {bucket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
