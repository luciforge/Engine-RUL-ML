"""Notification policy: service alert evaluation and threshold-cost simulation.

Two layers:
  1. Production trigger — evaluate_alert() converts model outputs into a
     structured ServiceAlert with urgency level and recommended service date.
     This drives the /schedule API endpoint.
  2. Simulation — sweep_threshold() and lead_time_cdf() sweep the decision
     threshold to quantify the operational cost/benefit trade-off.
     Used in notebooks/03_policy_simulation.ipynb.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Production trigger
# ---------------------------------------------------------------------------

@dataclass
class ServiceAlert:
    """Structured maintenance recommendation produced by evaluate_alert().

    Attributes
    ----------
    unit_id : int
    risk_score : float
        P(failure within label_within_x cycles) from the classifier.
    estimated_rul_cycles : float
        Predicted remaining useful life in operational cycles.
    urgency : str
        One of "critical" | "high" | "scheduled" | "ok".
    recommended_service_date : str
        ISO-8601 date derived from RUL and assumed cycles-per-day rate.
    days_until_service : int
        Calendar days to recommended service date.
    message : str
        Human-readable summary for client notification.
    """

    unit_id: int
    risk_score: float
    estimated_rul_cycles: float
    urgency: str
    recommended_service_date: str
    days_until_service: int
    message: str


def evaluate_alert(
    unit_id: int,
    risk_score: float,
    estimated_rul_cycles: float,
    threshold: float = 0.5,
    cycles_per_day: float = 3.0,
    reference_date: date | None = None,
) -> ServiceAlert:
    """Convert model outputs into a structured service alert.

    Urgency levels:
      critical  — risk_score >= 0.8  or  estimated_rul_cycles <= 10
      high      — risk_score >= threshold  (default 0.5)
      scheduled — risk_score >= 0.3  (early warning)
      ok        — below all thresholds

    Parameters
    ----------
    unit_id : int
    risk_score : float
        Classifier output from /predict.
    estimated_rul_cycles : float
        Regressor output from /predict.
    threshold : float
        Decision threshold for "high" urgency (matches API _THRESHOLD).
    cycles_per_day : float
        Assumed operational cycles per calendar day for date estimation.
    reference_date : date, optional
        Base date for service date calculation. Defaults to today.

    Returns
    -------
    ServiceAlert
    """
    if reference_date is None:
        reference_date = date.today()

    rul = max(0.0, float(estimated_rul_cycles))
    days = int(math.ceil(rul / max(cycles_per_day, 1e-9)))
    service_date = reference_date + timedelta(days=days)

    if risk_score >= 0.8 or rul <= 10:
        urgency = "critical"
        msg = (
            f"Unit {unit_id}: CRITICAL — immediate inspection required. "
            f"Estimated RUL {rul:.0f} cycles ({days} days). "
            f"Recommend service by {service_date.isoformat()}."
        )
    elif risk_score >= threshold:
        urgency = "high"
        msg = (
            f"Unit {unit_id}: HIGH RISK — schedule maintenance soon. "
            f"Estimated RUL {rul:.0f} cycles ({days} days). "
            f"Recommended service date: {service_date.isoformat()}."
        )
    elif risk_score >= 0.3:
        urgency = "scheduled"
        msg = (
            f"Unit {unit_id}: EARLY WARNING — plan service within schedule. "
            f"Estimated RUL {rul:.0f} cycles ({days} days). "
            f"Next service window: {service_date.isoformat()}."
        )
    else:
        urgency = "ok"
        msg = f"Unit {unit_id}: operating normally. Next check in {days} days."

    return ServiceAlert(
        unit_id=unit_id,
        risk_score=round(float(risk_score), 4),
        estimated_rul_cycles=round(rul, 1),
        urgency=urgency,
        recommended_service_date=service_date.isoformat(),
        days_until_service=days,
        message=msg,
    )


def sweep_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    rul: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep prediction threshold and compute operational metrics at each point.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth failure labels (1 = fail within X cycles).
    y_score : np.ndarray
        Predicted risk scores in [0, 1].
    rul : np.ndarray
        True RUL at each prediction point (for lead-time computation).
    thresholds : np.ndarray, optional
        Threshold values to sweep. Defaults to np.linspace(0.1, 0.9, 17).

    Returns
    -------
    pd.DataFrame with columns:
        threshold, alerts, missed_failures, prevented_failures,
        false_service_calls, mean_lead_time_cycles, median_lead_time_cycles.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    rows = []
    for t in thresholds:
        pred_pos = y_score >= t
        tp_mask = pred_pos & (y_true == 1)
        fn_mask = (~pred_pos) & (y_true == 1)
        fp_mask = pred_pos & (y_true == 0)

        lead_times = rul[tp_mask]
        mean_lead = float(lead_times.mean()) if len(lead_times) > 0 else 0.0
        median_lead = float(np.median(lead_times)) if len(lead_times) > 0 else 0.0

        rows.append(
            {
                "threshold": round(float(t), 3),
                "alerts": int(pred_pos.sum()),
                "missed_failures": int(fn_mask.sum()),
                "prevented_failures": int(tp_mask.sum()),
                "false_service_calls": int(fp_mask.sum()),
                "mean_lead_time_cycles": round(mean_lead, 2),
                "median_lead_time_cycles": round(median_lead, 2),
            }
        )

    return pd.DataFrame(rows)


def lead_time_cdf(
    y_true: np.ndarray,
    y_score: np.ndarray,
    rul: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return CDF of lead times for true positives at a given threshold."""
    pred_pos = y_score >= threshold
    tp_mask = pred_pos & (y_true == 1)
    lead_times = np.sort(rul[tp_mask])
    if len(lead_times) == 0:
        return pd.DataFrame({"lead_time_cycles": [], "cdf": []})
    cdf = np.arange(1, len(lead_times) + 1) / len(lead_times)
    return pd.DataFrame({"lead_time_cycles": lead_times, "cdf": cdf})
