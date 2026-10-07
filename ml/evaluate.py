"""Evaluation helpers for cross-validated classifier probabilities."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def choose_risk_thresholds(
    y_true, probabilities, method: str = "median out-of-fold probability"
) -> dict[str, float | str]:
    """Derive low/high display boundaries from negative/positive score medians.

    These are data-derived display bands, not clinical cutoffs or validated
    treatment thresholds. If score medians do not preserve the expected order,
    training stops instead of inventing risk boundaries.
    """
    import numpy as np

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    low = float(np.median(probabilities[y_true == 0]))
    high = float(np.median(probabilities[y_true == 1]))
    if not low < high:
        raise ValueError(
            "Score medians do not separate negative and positive "
            "classes; review model quality or supply justified risk thresholds."
        )
    return {
        "low_max": low,
        "high_min": high,
        "method": f"{method} among observed negative/positive labels",
        "clinical_cutoff": False,
    }


def confusion(y_true, probabilities, threshold: float = 0.5) -> list[list[int]]:
    return confusion_matrix(y_true, probabilities >= threshold, labels=[0, 1]).tolist()
