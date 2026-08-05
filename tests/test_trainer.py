"""End-to-end trainer test against a temp MLflow file store."""

from __future__ import annotations

from pathlib import Path

import pytest

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.ml.models import ModelType
from kepler_engine.ml.trainer import ExperimentTrainer
from kepler_engine.services.ingestion import LocalCSVDataSource
from kepler_engine.services.mlflow_client import MLflowService

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "samples" / "kepler_koi_sample.csv"


@pytest.fixture
def trainer_settings(tmp_mlflow, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("KEPLER_LOCAL_CSV_PATH", str(SAMPLE_CSV))
    monkeypatch.setenv("KEPLER_DATA_SOURCE", "local_csv")
    monkeypatch.setenv("KEPLER_PROMOTE_THRESHOLD", "0.0")
    monkeypatch.setenv("KEPLER_DEFAULT_CV_FOLDS", "2")
    get_settings.cache_clear()
    return get_settings()


def test_trainer_runs_random_forest(trainer_settings: Settings) -> None:
    trainer = ExperimentTrainer(
        settings=trainer_settings,
        data_source=LocalCSVDataSource(SAMPLE_CSV),
        mlflow_service=MLflowService(trainer_settings),
    )
    result = trainer.run(
        model_type=ModelType.RANDOM_FOREST,
        test_size=0.3,
        cv_folds=2,
        promote=True,
    )
    assert result["run_id"]
    assert "f1" in result["metrics"]
    assert result["model_version"] is not None
    assert result["promoted"] is True
    assert result["n_train"] > 0
    assert result["n_test"] > 0
