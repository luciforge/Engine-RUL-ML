"""Evaluation metrics scoped by task type.

Classification  (label_within_X):  PR-AUC, recall@precision=0.8, F1, Brier score, ECE
RUL regression:                     MAE, RMSE, asymmetric late-error penalty
Survival (CoxPH only):              C-index (via lifelines)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.8
) -> float:
    """Recall at the lowest threshold where precision >= target_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= target_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def asymmetric_rul_penalty(
    y_true: np.ndarray, y_pred: np.ndarray, late_weight: float = 1.5
) -> float:
    """Asymmetric loss: late predictions (y_pred > y_true) penalised at 1.5×.

    A positive error means the model over-estimated RUL (predicted too late),
    which is operationally worse — the engine may fail before maintenance.
    """
    errors = y_pred - y_true
    weights = np.where(errors > 0, late_weight, 1.0)
    return float(np.mean(weights * np.abs(errors)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(np.mean((y_prob - y_true) ** 2))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform binning)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece_val = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece_val += mask.sum() * abs(acc - conf)
    return float(ece_val / n)


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Full classification metric bundle (PR-AUC, recall@P80, F1, Brier, ECE)."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": pr_auc(y_true, y_score),
        "recall_at_p80": recall_at_precision(y_true, y_score, 0.8),
        "f1": f1(y_true, y_pred),
        "brier_score": brier_score(y_true, y_score),
        "ece": ece(y_true, y_score),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RUL regression metric bundle (MAE, RMSE, asymmetric penalty)."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "asymmetric_penalty": asymmetric_rul_penalty(y_true, y_pred),
    }


def business_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    rul: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> "pd.DataFrame":
    """Threshold-sweep business metrics table (wrapper around policy.notification.sweep_threshold).

    Returns a DataFrame with columns: threshold, alerts, missed_failures,
    prevented_failures, false_service_calls, mean_lead_time_cycles,
    median_lead_time_cycles.  Pass a cost_policy to also get expected_cost column.
    """
    import pandas as pd  # noqa: F401 — needed for return type at runtime
    from policy.notification import sweep_threshold
    return sweep_threshold(y_true, y_score, rul, thresholds)



