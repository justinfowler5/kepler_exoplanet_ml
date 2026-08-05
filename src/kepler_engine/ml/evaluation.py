"""Evaluation metrics and MLflow artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
    *,
    labels: list[str] | None = None,
    average: str = "binary",
    pos_label: str | int | None = None,
) -> dict[str, Any]:
    """Compute accuracy / precision / recall / f1 / roc_auc / pr_auc + report."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    unique = list(labels) if labels is not None else sorted(set(y_true_arr) | set(y_pred_arr))
    is_binary = len(unique) == 2
    avg = average if is_binary else "macro"
    kw: dict[str, Any] = {"average": avg, "zero_division": 0}
    if is_binary and pos_label is not None:
        kw["pos_label"] = pos_label

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, **kw)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, **kw)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, **kw)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=unique).tolist(),
        "classification_report": classification_report(
            y_true_arr, y_pred_arr, labels=unique, output_dict=True, zero_division=0
        ),
        "labels": unique,
    }

    if y_proba is not None:
        try:
            if is_binary:
                # y_proba may be (n, 2) or (n,). When (n, 2), use pos_label's column.
                if getattr(y_proba, "ndim", 1) == 2:
                    pos = pos_label if pos_label is not None else unique[-1]
                    pos_idx = list(unique).index(pos) if pos in unique else 1
                    proba = y_proba[:, pos_idx]
                else:
                    proba = y_proba
                    pos = pos_label if pos_label is not None else unique[-1]
                y_bin = (y_true_arr == pos).astype(int)
                metrics["roc_auc"] = float(roc_auc_score(y_bin, proba))
                metrics["pr_auc"] = float(average_precision_score(y_bin, proba))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true_arr, y_proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass

    return metrics


def save_confusion_matrix_png(
    cm: list[list[int]],
    labels: list[str],
    path: Path,
) -> Path:
    """Write a confusion-matrix heatmap PNG to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = arr.max() / 2.0 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                format(arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def compute_permutation_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on the raw feature columns (Pipeline as a whole)."""
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
