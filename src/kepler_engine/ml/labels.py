"""Label strategies for KOI disposition classification."""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

from kepler_engine.ml.features import TARGET_COLUMN


class LabelStrategy(StrEnum):
    """How to map ``koi_disposition`` into a training target."""

    BINARY = "binary"  # CONFIRMED vs FALSE POSITIVE (drop CANDIDATE)
    NOT_FALSE_POSITIVE = "not_false_positive"  # CONFIRMED|CANDIDATE vs FALSE POSITIVE
    MULTICLASS = "multiclass"  # three-way disposition


BINARY_POSITIVE = "CONFIRMED"
BINARY_NEGATIVE = "FALSE POSITIVE"
CANDIDATE = "CANDIDATE"
NOT_FALSE_POSITIVE = "NOT_FALSE_POSITIVE"

KNOWN_DISPOSITIONS: frozenset[str] = frozenset({BINARY_POSITIVE, BINARY_NEGATIVE, CANDIDATE})

_STRATEGY_KEEP: dict[LabelStrategy, frozenset[str]] = {
    LabelStrategy.BINARY: frozenset({BINARY_POSITIVE, BINARY_NEGATIVE}),
    LabelStrategy.NOT_FALSE_POSITIVE: KNOWN_DISPOSITIONS,
    LabelStrategy.MULTICLASS: KNOWN_DISPOSITIONS,
}


def apply_label_strategy(
    df: pd.DataFrame,
    strategy: LabelStrategy = LabelStrategy.BINARY,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter/map *df* and return ``(features_frame, y)``.

    The returned frame still contains all original columns; callers should
    select features separately. ``y`` is a Series of string labels (binary /
    multiclass) ready for sklearn.

    Rows whose disposition is not one of :data:`KNOWN_DISPOSITIONS` are dropped
    under every strategy. Malformed rows must never be silently folded into a
    class, which is what a catch-all "anything that isn't a false positive"
    mapping would do.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")

    try:
        strategy = LabelStrategy(strategy)
        keep = _STRATEGY_KEEP[strategy]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unknown label strategy: {strategy}") from exc

    disposition = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    mask = disposition.isin(keep)
    work = df.loc[mask].reset_index(drop=True)
    disposition = disposition.loc[mask].reset_index(drop=True)

    if strategy is LabelStrategy.NOT_FALSE_POSITIVE:
        y = disposition.where(disposition == BINARY_NEGATIVE, NOT_FALSE_POSITIVE)
    else:
        y = disposition

    return work, y.rename(TARGET_COLUMN)
