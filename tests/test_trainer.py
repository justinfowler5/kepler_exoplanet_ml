"""End-to-end trainer test against a temp MLflow file store."""

from __future__ import annotations

from pathlib import Path

import pytest

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.ml.features import FEATURE_COLUMNS
from kepler_engine.ml.labels import LabelStrategy
from kepler_engine.ml.models import ModelType
from kepler_engine.ml.trainer import ExperimentTrainer
from kepler_engine.services.inference_service import InferenceService
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


def _trainer(settings: Settings) -> ExperimentTrainer:
    return ExperimentTrainer(
        settings=settings,
        data_source=LocalCSVDataSource(SAMPLE_CSV),
        mlflow_service=MLflowService(settings),
    )


def test_trainer_runs_random_forest(trainer_settings: Settings) -> None:
    result = _trainer(trainer_settings).run(
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


@pytest.mark.parametrize(
    "model_type",
    [ModelType.LOGISTIC_REGRESSION, ModelType.GRADIENT_BOOSTING, ModelType.XGBOOST],
)
def test_trainer_supports_every_model_type(
    trainer_settings: Settings, model_type: ModelType
) -> None:
    result = _trainer(trainer_settings).run(
        model_type=model_type, test_size=0.3, cv_folds=2, promote=False
    )
    assert result["model_type"] == model_type.value
    assert 0.0 <= result["metrics"]["f1"] <= 1.0
    assert result["labels"] == ["CONFIRMED", "FALSE POSITIVE"]


@pytest.mark.parametrize(
    ("strategy", "expected_labels"),
    [
        (LabelStrategy.MULTICLASS, ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]),
        (LabelStrategy.NOT_FALSE_POSITIVE, ["FALSE POSITIVE", "NOT_FALSE_POSITIVE"]),
    ],
)
def test_trainer_label_strategies(
    trainer_settings: Settings, strategy: LabelStrategy, expected_labels: list[str]
) -> None:
    result = _trainer(trainer_settings).run(
        model_type=ModelType.GRADIENT_BOOSTING,
        label_strategy=strategy,
        test_size=0.3,
        cv_folds=2,
        promote=False,
    )
    assert result["labels"] == expected_labels


def test_promoted_champion_is_servable(trainer_settings: Settings) -> None:
    """The logged Pipeline must satisfy its own signature via models:/...@champion.

    Guards the training/serving dtype contract: the signature is all doubles, so
    integral request fields have to be coerced rather than rejected.
    """
    import pandas as pd

    result = _trainer(trainer_settings).run(
        model_type=ModelType.RANDOM_FOREST, test_size=0.3, cv_folds=2, promote=True
    )
    assert result["promoted"] is True

    records = pd.read_csv(SAMPLE_CSV)[FEATURE_COLUMNS].head(3).to_dict(orient="records")
    records[0]["koi_tce_plnt_num"] = 1  # python int, as FastAPI would deliver it

    predictions = InferenceService(
        settings=trainer_settings, mlflow_service=MLflowService(trainer_settings)
    ).predict(records)

    assert len(predictions) == 3
    for prediction in predictions:
        assert prediction["label"] in {"CONFIRMED", "FALSE POSITIVE"}
        assert 0.0 <= prediction["probability"] <= 1.0
        assert prediction["model_version"] == result["model_version"]
