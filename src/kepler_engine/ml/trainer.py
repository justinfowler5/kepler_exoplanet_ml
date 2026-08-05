"""Experiment trainer: ingest -> validate -> label -> fit -> evaluate -> MLflow."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from kepler_engine.core.config import Settings, get_settings
from kepler_engine.core.logging import get_logger
from kepler_engine.ml.evaluation import (
    compute_metrics,
    compute_permutation_importance,
    save_confusion_matrix_png,
)
from kepler_engine.ml.features import FEATURE_COLUMNS, assert_no_leakage, select_feature_matrix
from kepler_engine.ml.labels import BINARY_POSITIVE, LabelStrategy, apply_label_strategy
from kepler_engine.ml.models import ModelType, get_estimator, needs_scaling
from kepler_engine.ml.preprocessing import build_preprocessor
from kepler_engine.services.ingestion import KeplerDataSource, get_data_source
from kepler_engine.services.mlflow_client import MLflowService

logger = get_logger(__name__)


class ExperimentTrainer:
    """Orchestrates a single training experiment end to end."""

    def __init__(
        self,
        settings: Settings | None = None,
        data_source: KeplerDataSource | None = None,
        mlflow_service: MLflowService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.data_source = data_source or get_data_source(self.settings)
        self.mlflow_service = mlflow_service or MLflowService(self.settings)

    def run(
        self,
        *,
        model_type: ModelType | str = ModelType.RANDOM_FOREST,
        hyperparams: dict[str, Any] | None = None,
        test_size: float | None = None,
        cv_folds: int | None = None,
        label_strategy: LabelStrategy | str = LabelStrategy.BINARY,
        promote: bool = True,
    ) -> dict[str, Any]:
        settings = self.settings
        test_size = test_size if test_size is not None else settings.default_test_size
        cv_folds = cv_folds if cv_folds is not None else settings.default_cv_folds
        strategy = LabelStrategy(label_strategy)
        mt = ModelType(model_type)

        logger.info("experiment.start", model_type=mt.value, label_strategy=strategy.value)

        raw = self.data_source.load()
        # Raw KOI tables contain leakage columns; we allowlist features instead of
        # rejecting the file. assert_no_leakage runs on the selected feature matrix.
        labeled, y = apply_label_strategy(raw, strategy)
        X = select_feature_matrix(labeled)
        assert_no_leakage(X)

        class_names = sorted(y.astype(str).unique())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=settings.random_state,
            stratify=y,
        )

        estimator, resolved_params = get_estimator(
            mt, hyperparams, random_state=settings.random_state
        )
        preprocessor = build_preprocessor(scale=needs_scaling(mt))
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        mlflow.sklearn.autolog(log_models=False, silent=True)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "model_type": mt.value,
                    "label_strategy": strategy.value,
                    "test_size": test_size,
                    "cv_folds": cv_folds,
                    "n_features": len(FEATURE_COLUMNS),
                    "n_rows": len(X),
                    **{f"hp_{k}": v for k, v in resolved_params.items()},
                }
            )

            if cv_folds and cv_folds > 1:
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=settings.random_state
                )
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
                mlflow.log_metric("cv_f1_macro_mean", float(cv_scores.mean()))
                mlflow.log_metric("cv_f1_macro_std", float(cv_scores.std()))

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_test)
                except Exception:  # noqa: BLE001
                    y_proba = None

            pos_label = BINARY_POSITIVE if BINARY_POSITIVE in class_names else class_names[-1]
            average = "binary" if len(class_names) == 2 else "macro"
            proba_for_metrics = y_proba
            if y_proba is not None and len(class_names) == 2 and getattr(y_proba, "ndim", 1) == 2:
                # LabelDecodingClassifier / LabelEncoder order matches sorted class_names
                pos_idx = class_names.index(pos_label)
                # But predict_proba columns follow encoder.classes_ order (= sorted)
                encoder_classes = list(
                    pipeline.named_steps["model"].encoder_.classes_
                )
                pos_idx = encoder_classes.index(pos_label)
                proba_for_metrics = y_proba[:, pos_idx]
            metrics = compute_metrics(
                y_test,
                y_pred,
                proba_for_metrics,
                labels=class_names,
                average=average,
                pos_label=pos_label if len(class_names) == 2 else None,
            )

            scalar_keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
            for key in scalar_keys:
                if key in metrics:
                    mlflow.log_metric(key, float(metrics[key]))

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                cm_path = save_confusion_matrix_png(
                    metrics["confusion_matrix"],
                    metrics["labels"],
                    tmp_path / "confusion_matrix.png",
                )
                mlflow.log_artifact(str(cm_path))

                try:
                    importance = compute_permutation_importance(
                        pipeline,
                        X_test,
                        y_test,
                        random_state=settings.random_state,
                    )
                    imp_path = tmp_path / "permutation_importance.csv"
                    importance.to_csv(imp_path, index=False)
                    mlflow.log_artifact(str(imp_path))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("permutation_importance.failed", error=str(exc))

            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
                input_example=X_train.head(3),
                registered_model_name=None,
            )

            # Persist label encoder classes as a tag for inference decoding.
            mlflow.set_tag("label_classes", ",".join(class_names))
            mlflow.set_tag("feature_columns", ",".join(FEATURE_COLUMNS))

            model_version: str | None = None
            promoted = False
            promote_metric = settings.promote_metric
            score = float(metrics.get(promote_metric, metrics.get("f1", 0.0)))
            if promote and score >= settings.promote_threshold:
                model_version = self.mlflow_service.register_and_promote(
                    model_uri=model_info.model_uri,
                    run_id=run_id,
                )
                promoted = True
            elif promote:
                model_version = self.mlflow_service.register_model(
                    model_uri=model_info.model_uri,
                    run_id=run_id,
                )

            result = {
                "run_id": run_id,
                "model_type": mt.value,
                "label_strategy": strategy.value,
                "metrics": {k: metrics[k] for k in scalar_keys if k in metrics},
                "confusion_matrix": metrics["confusion_matrix"],
                "labels": class_names,
                "model_uri": model_info.model_uri,
                "model_version": model_version,
                "promoted": promoted,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }
            logger.info(
                "experiment.complete",
                run_id=run_id,
                f1=metrics.get("f1"),
                promoted=promoted,
                model_version=model_version,
            )
            return result
