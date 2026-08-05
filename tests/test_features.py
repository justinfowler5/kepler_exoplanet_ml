"""Leakage-guard tests — highest-value suite."""

from __future__ import annotations

import pandas as pd
import pytest

from kepler_engine.core.exceptions import LeakageViolationError
from kepler_engine.ml.features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    assert_no_leakage,
    select_feature_matrix,
)


def test_feature_allowlist_size() -> None:
    assert len(FEATURE_COLUMNS) == 14


@pytest.mark.parametrize("col", sorted(LEAKAGE_COLUMNS))
def test_each_leakage_column_is_rejected(sample_df: pd.DataFrame, col: str) -> None:
    # Ensure the column is present, then assert rejection.
    if col not in sample_df.columns:
        sample_df = sample_df.copy()
        sample_df[col] = 0
    with pytest.raises(LeakageViolationError) as exc:
        assert_no_leakage(sample_df)
    assert col in str(exc.value)


def test_clean_feature_matrix_passes(feature_only_df: pd.DataFrame) -> None:
    assert_no_leakage(feature_only_df)
    X = select_feature_matrix(feature_only_df)
    assert list(X.columns) == FEATURE_COLUMNS


def test_select_rejects_missing_required_column(feature_only_df: pd.DataFrame) -> None:
    df = feature_only_df.drop(columns=["koi_period"])
    with pytest.raises(LeakageViolationError, match="missing"):
        select_feature_matrix(df)


def test_fpflags_in_denylist() -> None:
    for flag in ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"):
        assert flag in LEAKAGE_COLUMNS
    assert "koi_score" in LEAKAGE_COLUMNS
    assert "koi_pdisposition" in LEAKAGE_COLUMNS
