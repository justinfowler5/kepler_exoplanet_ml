"""Feature allowlist / leakage denylist — the primary correctness guard."""

from __future__ import annotations

import pandas as pd

from kepler_engine.core.exceptions import LeakageViolationError

TARGET_COLUMN = "koi_disposition"

FEATURE_COLUMNS: list[str] = [
    "koi_period",
    "koi_time0bk",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_tce_plnt_num",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

# Columns that reproduce or restate the vetting decision. Including any of these
# yields ~99% "accuracy" that collapses on unvetted KOIs.
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "koi_pdisposition",
        "koi_score",
        "kepler_name",
        "koi_comment",
        "koi_disp_prov",
        "koi_vet_stat",
        "koi_vet_date",
    }
)

# Transit-fit block that is missing together on ~363 rows.
TRANSIT_FIT_COLUMNS: list[str] = [
    "koi_impact",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise if *df* contains any known leakage column among its feature set.

    The target column itself is allowed to be present (labels need it); every
    other denylisted column is forbidden.
    """
    present = set(df.columns) & LEAKAGE_COLUMNS
    if present:
        raise LeakageViolationError(
            f"Training frame contains leakage columns: {sorted(present)}. "
            "These restate the Robovetter decision and must not be used as features."
        )


def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the allowlisted feature columns in canonical order."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise LeakageViolationError(f"Required feature columns missing: {missing}")
    assert_no_leakage(df[FEATURE_COLUMNS])
    return df[FEATURE_COLUMNS].copy()
