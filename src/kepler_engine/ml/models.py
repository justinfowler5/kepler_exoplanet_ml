"""Estimator factory / model registry."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"


DEFAULT_PARAMS: dict[ModelType, dict[str, Any]] = {
    ModelType.LOGISTIC_REGRESSION: {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
    },
    ModelType.RANDOM_FOREST: {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    ModelType.GRADIENT_BOOSTING: {
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 4,
    },
    ModelType.XGBOOST: {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "n_jobs": -1,
    },
}


class LabelDecodingClassifier(ClassifierMixin, BaseEstimator):
    """Wraps an estimator so ``predict`` returns original string labels."""

    def __init__(self, estimator: Any | None = None) -> None:
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.encoder_ = LabelEncoder().fit(y)
        y_enc = self.encoder_.transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_enc)
        return self

    def predict(self, X):
        pred = self.estimator_.predict(X)
        return self.encoder_.inverse_transform(pred)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def get_estimator(
    model_type: ModelType | str,
    hyperparams: dict[str, Any] | None = None,
    *,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(estimator, resolved_params)`` for *model_type*."""
    mt = ModelType(model_type)
    params = {**DEFAULT_PARAMS[mt], **(hyperparams or {})}
    params.setdefault("random_state", random_state)

    if mt is ModelType.LOGISTIC_REGRESSION:
        base: Any = LogisticRegression(**params)
    elif mt is ModelType.RANDOM_FOREST:
        base = RandomForestClassifier(**params)
    elif mt is ModelType.GRADIENT_BOOSTING:
        base = GradientBoostingClassifier(**params)
    elif mt is ModelType.XGBOOST:
        from xgboost import XGBClassifier

        base = XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {mt}")

    # Always wrap so /predict receives string disposition labels.
    estimator = LabelDecodingClassifier(estimator=base)
    return estimator, params


def needs_scaling(model_type: ModelType | str) -> bool:
    return ModelType(model_type) is ModelType.LOGISTIC_REGRESSION
