"""Leakage-guard tests — highest-value suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kepler_engine.core.exceptions import LeakageViolationError
from kepler_engine.ml.features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    TRANSIT_FIT_COLUMNS,
    assert_no_leakage,
    select_feature_matrix,
)
from kepler_engine.ml.labels import BINARY_NEGATIVE, LabelStrategy, apply_label_strategy
from kepler_engine.ml.preprocessing import build_preprocessor


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


def test_allowlist_and_denylist_are_disjoint() -> None:
    assert not set(FEATURE_COLUMNS) & LEAKAGE_COLUMNS


def test_leakage_detection_ignores_header_casing(feature_only_df: pd.DataFrame) -> None:
    df = feature_only_df.assign(KOI_Score=0.0)
    with pytest.raises(LeakageViolationError, match="KOI_Score"):
        assert_no_leakage(df)


@pytest.mark.parametrize("strategy", list(LabelStrategy))
def test_unknown_dispositions_are_dropped(sample_df: pd.DataFrame, strategy: LabelStrategy) -> None:
    _, baseline = apply_label_strategy(sample_df, strategy)
    assert sample_df.loc[[0, 1], TARGET_COLUMN].eq("CONFIRMED").all()

    df = sample_df.copy()
    df.loc[0, TARGET_COLUMN] = "0"
    df.loc[1, TARGET_COLUMN] = None

    _, y = apply_label_strategy(df, strategy)

    assert len(y) == len(baseline) - 2
    assert not {"0", "NONE", "NAN"} & set(y)


def test_not_false_positive_does_not_absorb_junk_rows(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy()
    df.loc[0, TARGET_COLUMN] = "0"

    kept, y = apply_label_strategy(df, LabelStrategy.NOT_FALSE_POSITIVE)

    assert set(y) <= {BINARY_NEGATIVE, "NOT_FALSE_POSITIVE"}
    assert "0" not in set(kept[TARGET_COLUMN])


def test_preprocessor_coerces_unparseable_values(feature_only_df: pd.DataFrame) -> None:
    df = feature_only_df.astype({"koi_prad": object})
    df.loc[0, "koi_prad"] = "N/A"

    out = build_preprocessor().fit_transform(select_feature_matrix(df))

    assert out.shape == (len(df), len(FEATURE_COLUMNS) + 1)
    assert np.isfinite(out).all()


def test_block_missing_indicator_flags_absent_transit_fit(
    feature_only_df: pd.DataFrame,
) -> None:
    X = select_feature_matrix(feature_only_df)
    expected = X[TRANSIT_FIT_COLUMNS].isna().all(axis=1)
    assert expected.sum() == 1, "fixture should retain one fully-missing transit-fit row"

    out = build_preprocessor().fit_transform(X)

    assert out[:, -1].tolist() == expected.astype(float).tolist()
