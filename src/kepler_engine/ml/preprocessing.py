"""Preprocessing: median imputation, blockwise missing indicator, optional scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from kepler_engine.ml.features import FEATURE_COLUMNS, TRANSIT_FIT_COLUMNS


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Coerce feature columns to float, turning unparseable values into NaN.

    KOI exports occasionally carry sentinel or shifted text in numeric fields.
    Without this the median imputer raises instead of treating them as missing,
    which would surface as a 500 on ``/predict`` rather than a low-confidence
    answer.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def _resolve(self) -> list[str]:
        return list(self.columns) if self.columns is not None else list(FEATURE_COLUMNS)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns = self._resolve()
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=columns)
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float64)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self._resolve())


class BlockMissingIndicator(BaseEstimator, TransformerMixin):
    """Single flag: True when the entire transit-fit block is missing."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def _resolve(self) -> list[str]:
        return list(self.columns) if self.columns is not None else list(TRANSIT_FIT_COLUMNS)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            block = X[self._resolve()]
            flag = block.isna().all(axis=1).astype(np.float64).to_numpy()
        else:
            flag = np.isnan(np.asarray(X, dtype=float)).all(axis=1).astype(np.float64)
        return flag.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["transit_fit_missing"])


def build_preprocessor(*, scale: bool = False) -> Pipeline:
    """Build the feature preprocessor used inside the training Pipeline.

    - Numeric coercion so unparseable values become missing rather than errors.
    - Median imputation for all feature columns.
    - A single blockwise MissingIndicator over the transit-fit group.
    - Optional StandardScaler (for linear models).
    """
    impute_all = ColumnTransformer(
        transformers=[
            ("impute", SimpleImputer(strategy="median"), FEATURE_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    transit_missing = ColumnTransformer(
        transformers=[
            ("transit_missing", BlockMissingIndicator(TRANSIT_FIT_COLUMNS), TRANSIT_FIT_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    union = FeatureUnion(
        transformer_list=[
            ("imputed", impute_all),
            ("missing", transit_missing),
        ]
    )

    steps: list[tuple[str, object]] = [
        ("coerce", NumericCoercer(FEATURE_COLUMNS)),
        ("features", union),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)
