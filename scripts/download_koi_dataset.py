#!/usr/bin/env python3
"""Download the Kepler KOI cumulative table from NASA TAP into data/raw."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import httpx

DEFAULT_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+*+from+cumulative&format=csv"
)
DEFAULT_OUT = Path("data/raw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    out_path = args.out_dir / f"kepler_koi_cumulative_{stamp}.csv"

    print(f"Fetching {args.url}")
    with httpx.stream("GET", args.url, timeout=180.0, follow_redirects=True) as response:
        response.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
