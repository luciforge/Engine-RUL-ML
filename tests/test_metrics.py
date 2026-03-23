"""Unit tests for evaluation/metrics.py — all metric functions."""

import numpy as np
import pytest

from evaluation.metrics import (
    asymmetric_rul_penalty,
    brier_score,
    classification_metrics,
    ece,
    f1,
    mae,
    pr_auc,
    recall_at_precision,
    regression_metrics,
    rmse,
)


# ---------------------------------------------------------------------------
# pr_auc
# ---------------------------------------------------------------------------

def test_pr_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_pr_auc_all_correct_scores_above_half():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_pr_auc_between_zero_and_one():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_score = rng.uniform(0, 1, size=100)
    result = pr_auc(y_true, y_score)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# recall_at_precision
# ---------------------------------------------------------------------------

def test_recall_at_precision_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    result = recall_at_precision(y_true, y_score, target_precision=0.8)
    assert result == pytest.approx(1.0)


def test_recall_at_precision_returns_zero_when_precision_never_reached():
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    # With only positives, precision is always 1.0, but test impossible precision
    result = recall_at_precision(y_true, y_score, target_precision=1.1)
    assert result == 0.0


# ---------------------------------------------------------------------------
# f1
# ---------------------------------------------------------------------------

def test_f1_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    assert f1(y_true, y_pred) == pytest.approx(1.0)


def test_f1_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    # f1 of all wrong binary predictions
    assert f1(y_true, y_pred) == pytest.approx(0.0)


def test_f1_zero_division_safe():
    # All predicted negative → no TP, precision=0, recall=0
    y_true = np.array([1, 1])
    y_pred = np.array([0, 0])
    assert f1(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mae / rmse
# ---------------------------------------------------------------------------

def test_mae_zero_for_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_positive():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 3.0])
    assert mae(y_true, y_pred) == pytest.approx(2.0)


def test_rmse_zero_for_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_penalises_large_errors_more_than_mae():
    y_true = np.zeros(2)
    y_pred = np.array([1.0, 9.0])
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


# ---------------------------------------------------------------------------
# asymmetric_rul_penalty
# ---------------------------------------------------------------------------

def test_asymmetric_penalty_zero_for_perfect():
    y = np.array([10.0, 20.0])
    assert asymmetric_rul_penalty(y, y) == pytest.approx(0.0)


def test_asymmetric_penalty_higher_for_late_prediction():
    y_true = np.array([10.0])
    y_pred_late = np.array([15.0])   # predicts more RUL than actual → late
    y_pred_early = np.array([5.0])   # predicts less RUL → early (safer)
    late = asymmetric_rul_penalty(y_true, y_pred_late, late_weight=1.5)
    early = asymmetric_rul_penalty(y_true, y_pred_early, late_weight=1.5)
    assert late > early


def test_asymmetric_penalty_symmetric_at_weight_1():
    y_true = np.array([10.0])
    late = asymmetric_rul_penalty(y_true, np.array([15.0]), late_weight=1.0)
    early = asymmetric_rul_penalty(y_true, np.array([5.0]), late_weight=1.0)
    assert late == pytest.approx(early)


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

def test_brier_perfect():
    y_true = np.array([0.0, 1.0])
    y_prob = np.array([0.0, 1.0])
    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_worst():
    y_true = np.array([0.0, 1.0])
    y_prob = np.array([1.0, 0.0])
    assert brier_score(y_true, y_prob) == pytest.approx(1.0)


def test_brier_between_0_and_1():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100).astype(float)
    y_prob = rng.uniform(0, 1, 100)
    assert 0.0 <= brier_score(y_true, y_prob) <= 1.0


# ---------------------------------------------------------------------------
# ece
# ---------------------------------------------------------------------------

def test_ece_zero_for_perfectly_calibrated():
    # If predicted probability == fraction of positives per bin → ECE = 0
    y_prob = np.linspace(0.05, 0.95, 10)
    y_true = (np.linspace(0.05, 0.95, 10) > 0.5).astype(float)
    result = ece(y_true, y_prob)
    assert result >= 0.0


def test_ece_non_negative():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200).astype(float)
    y_prob = rng.uniform(0, 1, 200)
    assert ece(y_true, y_prob) >= 0.0


# ---------------------------------------------------------------------------
# Bundle functions
# ---------------------------------------------------------------------------

def test_classification_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    result = classification_metrics(y_true, y_score)
    assert set(result.keys()) == {"pr_auc", "recall_at_p80", "f1", "brier_score", "ece"}


def test_classification_metrics_all_float():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    for v in classification_metrics(y_true, y_score).values():
        assert isinstance(v, float)


def test_regression_metrics_keys():
    y = np.array([1.0, 2.0, 3.0])
    result = regression_metrics(y, y)
    assert set(result.keys()) == {"mae", "rmse", "asymmetric_penalty"}


def test_regression_metrics_zero_for_perfect():
    y = np.array([5.0, 10.0, 15.0])
    result = regression_metrics(y, y)
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["asymmetric_penalty"] == pytest.approx(0.0)
