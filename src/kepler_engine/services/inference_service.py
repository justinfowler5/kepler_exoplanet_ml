"""Cached champion-model inference service."""

from __future__ import annotations

import threading
import time
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.core.exceptions import ModelNotFoundError
from kepler_engine.core.logging import get_logger
from kepler_engine.ml.features import FEATURE_COLUMNS
from kepler_engine.services.mlflow_client import MLflowService

logger = get_logger(__name__)


class InferenceService:
    """Loads ``models:/...@champion`` via mlflow.pyfunc with a TTL cache."""

    def __init__(
        self,
        settings: Settings | None = None,
        mlflow_service: MLflowService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.mlflow_service = mlflow_service or MLflowService(self.settings)
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._loaded_at: float = 0.0
        self._model_version: str | None = None
        self._model_uri: str | None = None

    def _cache_valid(self) -> bool:
        if self._model is None:
            return False
        return (time.monotonic() - self._loaded_at) < self.settings.model_cache_ttl_seconds

    def _load(self) -> None:
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        uri = self.mlflow_service.champion_model_uri()
        try:
            model = mlflow.pyfunc.load_model(uri)
            version = self.mlflow_service.resolve_champion_version()
        except Exception as exc:  # noqa: BLE001
            raise ModelNotFoundError(f"Failed to load champion model at {uri}: {exc}") from exc
        self._model = model
        self._model_uri = uri
        self._model_version = version
        self._loaded_at = time.monotonic()
        logger.info("inference.model_loaded", uri=uri, version=version)

    def warm(self) -> bool:
        """Attempt to warm the cache; return False if no champion exists yet."""
        try:
            with self._lock:
                self._load()
            return True
        except ModelNotFoundError:
            logger.warning("inference.warm_skipped", reason="no champion model")
            return False

    def get_model(self) -> Any:
        with self._lock:
            if not self._cache_valid():
                self._load()
            return self._model

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def records_to_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        frame = df[FEATURE_COLUMNS].copy()
        # Training casts every feature to float64, so the logged signature is all
        # doubles. Pandas infers int64 for integral fields such as koi_tce_plnt_num,
        # and MLflow rejects that widening rather than performing it.
        for column in FEATURE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        return frame

    def _unwrap_sklearn(self, model: Any) -> Any | None:
        unwrapped = getattr(model, "_model_impl", None)
        if unwrapped is None:
            return None
        return getattr(unwrapped, "sklearn_model", None) or getattr(
            unwrapped, "_model_impl", None
        )

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        model = self.get_model()
        X = self.records_to_dataframe(records)
        raw_preds = model.predict(X)

        probabilities: list[float | None] = [None] * len(records)
        predictor = self._unwrap_sklearn(model)
        class_names: list[str] | None = None
        if predictor is not None:
            est = predictor.named_steps.get("model", predictor) if hasattr(
                predictor, "named_steps"
            ) else predictor
            encoder = getattr(est, "encoder_", None)
            if encoder is not None:
                class_names = [str(c) for c in encoder.classes_]
            if hasattr(predictor, "predict_proba"):
                try:
                    proba = predictor.predict_proba(X)
                    if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
                        conf_idx = 0
                        if class_names and "CONFIRMED" in class_names:
                            conf_idx = class_names.index("CONFIRMED")
                        probabilities = [float(p) for p in proba[:, conf_idx]]
                    else:
                        probabilities = [float(p) for p in proba]
                except Exception:  # noqa: BLE001
                    pass

        results: list[dict[str, Any]] = []
        for i, pred in enumerate(raw_preds):
            label = str(pred)
            results.append(
                {
                    "label": label,
                    "probability": probabilities[i],
                    "model_version": self._model_version,
                    "run_id": None,
                }
            )
        return results

    def is_loadable(self) -> bool:
        try:
            self.get_model()
            return True
        except ModelNotFoundError:
            return False
