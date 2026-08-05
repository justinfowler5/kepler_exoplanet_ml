"""Label strategies for KOI disposition classification."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from kepler_engine.ml.features import TARGET_COLUMN


class LabelStrategy(str, Enum):
    """How to map ``koi_disposition`` into a training target."""

    BINARY = "binary"  # CONFIRMED vs FALSE POSITIVE (drop CANDIDATE)
    NOT_FALSE_POSITIVE = "not_false_positive"  # CONFIRMED|CANDIDATE vs FALSE POSITIVE
    MULTICLASS = "multiclass"  # three-way disposition


BINARY_POSITIVE = "CONFIRMED"
BINARY_NEGATIVE = "FALSE POSITIVE"


def apply_label_strategy(
    df: pd.DataFrame,
    strategy: LabelStrategy = LabelStrategy.BINARY,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter/map *df* and return ``(features_frame, y)``.

    The returned frame still contains all original columns; callers should
    select features separately. ``y`` is a Series of string labels (binary /
    multiclass) ready for sklearn.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")

    work = df.copy()
    disposition = work[TARGET_COLUMN].astype(str).str.strip().str.upper()

    if strategy is LabelStrategy.BINARY:
        mask = disposition.isin({BINARY_POSITIVE, BINARY_NEGATIVE})
        work = work.loc[mask].reset_index(drop=True)
        y = (
            work[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({BINARY_POSITIVE: BINARY_POSITIVE, BINARY_NEGATIVE: BINARY_NEGATIVE})
        )
    elif strategy is LabelStrategy.NOT_FALSE_POSITIVE:
        y = disposition.map(
            lambda d: BINARY_NEGATIVE if d == BINARY_NEGATIVE else "NOT_FALSE_POSITIVE"
        )
        work = work.reset_index(drop=True)
        y = y.reset_index(drop=True)
    elif strategy is LabelStrategy.MULTICLASS:
        allowed = {"CONFIRMED", "FALSE POSITIVE", "CANDIDATE"}
        mask = disposition.isin(allowed)
        work = work.loc[mask].reset_index(drop=True)
        y = work[TARGET_COLUMN].astype(str).str.strip().str.upper()
    else:
        raise ValueError(f"Unknown label strategy: {strategy}")

    return work, y
