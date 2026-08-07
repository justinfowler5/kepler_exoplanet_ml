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


_ALLOWLIST_LEAK_OVERLAP = sorted(set(FEATURE_COLUMNS) & LEAKAGE_COLUMNS)
if _ALLOWLIST_LEAK_OVERLAP:
    raise LeakageViolationError(
        f"FEATURE_COLUMNS declares leakage columns as features: {_ALLOWLIST_LEAK_OVERLAP}"
    )


def _normalize(name: object) -> str:
    return str(name).strip().lower()


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise if *df* contains any known leakage column among its feature set.

    The target column itself is allowed to be present (labels need it); every
    other denylisted column is forbidden. Matching ignores case and surrounding
    whitespace because NASA TAP exports do not guarantee header casing.
    """
    present = sorted(c for c in df.columns if _normalize(c) in LEAKAGE_COLUMNS)
    if present:
        raise LeakageViolationError(
            f"Training frame contains leakage columns: {present}. "
            "These restate the Robovetter decision and must not be used as features."
        )


def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the allowlisted feature columns in canonical order, as float64.

    Every allowlisted feature is a physical measurement, so the matrix is
    numeric by definition. Forcing float64 also keeps the logged MLflow
    signature free of integer columns, which would otherwise reject nulls at
    inference time for sparsely populated fields like ``koi_tce_plnt_num``.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise LeakageViolationError(f"Required feature columns missing: {missing}")
    selected = df[FEATURE_COLUMNS].copy()
    assert_no_leakage(selected)
    for column in FEATURE_COLUMNS:
        selected[column] = pd.to_numeric(selected[column], errors="coerce").astype("float64")
    return selected
