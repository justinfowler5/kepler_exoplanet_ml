"""Thin MLflow 3 wrapper for registration, aliases, and run listing."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.core.exceptions import ModelNotFoundError
from kepler_engine.core.logging import get_logger

logger = get_logger(__name__)


class MLflowService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        self._client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)

    @property
    def client(self) -> MlflowClient:
        return self._client

    def register_model(self, *, model_uri: str, run_id: str) -> str:
        name = self.settings.registered_model_name
        result = mlflow.register_model(model_uri, name)
        logger.info("mlflow.registered", name=name, version=result.version, run_id=run_id)
        return str(result.version)

    def set_champion_alias(self, version: str | int) -> None:
        name = self.settings.registered_model_name
        alias = self.settings.model_alias
        self._client.set_registered_model_alias(name, alias, str(version))
        logger.info("mlflow.alias_set", name=name, alias=alias, version=str(version))

    def register_and_promote(self, *, model_uri: str, run_id: str) -> str:
        version = self.register_model(model_uri=model_uri, run_id=run_id)
        self.set_champion_alias(version)
        return version

    def champion_model_uri(self) -> str:
        name = self.settings.registered_model_name
        alias = self.settings.model_alias
        return f"models:/{name}@{alias}"

    def resolve_champion_version(self) -> str:
        name = self.settings.registered_model_name
        alias = self.settings.model_alias
        try:
            mv = self._client.get_model_version_by_alias(name, alias)
            return str(mv.version)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotFoundError(
                f"No model version with alias '{alias}' for '{name}'"
            ) from exc

    def latest_version(self) -> str | None:
        name = self.settings.registered_model_name
        versions = self._client.search_model_versions(
            filter_string=f"name='{name}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if not versions:
            return None
        return str(versions[0].version)

    def list_recent_runs(self, max_results: int = 20) -> list[dict[str, Any]]:
        experiment = self._client.get_experiment_by_name(self.settings.mlflow_experiment_name)
        if experiment is None:
            return []
        runs = self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=max_results,
        )
        out: list[dict[str, Any]] = []
        for run in runs:
            out.append(
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags),
                }
            )
        return out

    def ping(self) -> bool:
        try:
            self._client.search_experiments(max_results=1)
            return True
        except Exception:  # noqa: BLE001
            return False
