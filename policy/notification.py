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
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cost_cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("policy", {})


# ---------------------------------------------------------------------------
# Cost-sensitive decision policy
# ---------------------------------------------------------------------------

@dataclass
class CostPolicy:
    """Economic cost parameters for the Replace / Inspect / Continue decision.

    All values in a consistent currency unit (e.g. EUR).

    Attributes
    ----------
    cost_replacement      : cost of a scheduled component replacement
    cost_unplanned_failure: cost of an unplanned failure (downtime + repair)
    cost_inspection       : cost of a targeted inspection visit
    cost_false_alarm      : wasted cost when unit is healthy but inspected
    """

    cost_replacement: float = 8000.0
    cost_unplanned_failure: float = 50000.0
    cost_inspection: float = 400.0
    cost_false_alarm: float = 200.0

    @classmethod
    def from_config(cls) -> "CostPolicy":
        """Load cost parameters from the project config.yaml policy section."""
        cfg = _cost_cfg()
        return cls(
            cost_replacement=float(cfg.get("cost_replacement", 8000.0)),
            cost_unplanned_failure=float(cfg.get("cost_unplanned_failure", 50000.0)),
            cost_inspection=float(cfg.get("cost_inspection", 400.0)),
            cost_false_alarm=float(cfg.get("cost_false_alarm", 200.0)),
        )


def expected_cost_action(
    risk: float,
    rul: float,
    policy: CostPolicy,
) -> dict:
    """Compute expected cost for each maintenance action and return the optimal one.

    Three possible actions:

    * **replace** — schedule a replacement now.
      Expected cost = cost_replacement (certain).

    * **inspect** — perform a targeted inspection.
      If failure imminent (risk >= 0.5), inspection leads to replacement anyway
      (cost_replacement + cost_inspection).  Otherwise just cost_inspection.

    * **continue** — take no action.
      Expected cost = risk * cost_unplanned_failure  (probabilistic failure cost)
                    + (1 - risk) * 0  (no cost if unit stays healthy).
      When rul <= 0 this converges to cost_unplanned_failure.

    Parameters
    ----------
    risk   : float in [0, 1] — predicted failure probability
    rul    : float >= 0     — estimated remaining useful life (cycles)
    policy : CostPolicy     — cost parameters

    Returns
    -------
    dict with keys:
        ``replace`` (float), ``inspect`` (float), ``continue`` (float),
        ``action`` (str — optimal action name),
        ``expected_cost`` (float — cost of the optimal action)
    """
    risk = float(np.clip(risk, 0.0, 1.0))
    rul = max(0.0, float(rul))

    c_replace = policy.cost_replacement
    # Inspection: if unit needs replacement after inspect, add replacement cost
    c_inspect = policy.cost_inspection + risk * policy.cost_replacement + (1.0 - risk) * policy.cost_false_alarm
    c_continue = risk * policy.cost_unplanned_failure

    costs = {"replace": c_replace, "inspect": c_inspect, "continue": c_continue}
    optimal_action = min(costs, key=costs.__getitem__)

    return {
        **costs,
        "action": optimal_action,
        "expected_cost": costs[optimal_action],
    }


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
    action : str
        Optimal cost-policy action: "replace" | "inspect" | "continue".
    expected_cost : float
        Expected cost (EUR) of the recommended action.
    """

    unit_id: int
    risk_score: float
    estimated_rul_cycles: float
    urgency: str
    recommended_service_date: str
    days_until_service: int
    message: str
    action: str = field(default="")
    expected_cost: float = field(default=0.0)


def evaluate_alert(
    unit_id: int,
    risk_score: float,
    estimated_rul_cycles: float,
    threshold: float = 0.5,
    cycles_per_day: float = 3.0,
    reference_date: date | None = None,
    cost_policy: CostPolicy | None = None,
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
        Decision threshold for "high" urgency (matches API risk_threshold config).
    cycles_per_day : float
        Assumed operational cycles per calendar day for date estimation.
    reference_date : date, optional
        Base date for service date calculation. Defaults to today.
    cost_policy : CostPolicy, optional
        If provided, computes the economically optimal action and expected cost.
        Defaults to CostPolicy.from_config().

    Returns
    -------
    ServiceAlert
    """
    if reference_date is None:
        reference_date = date.today()

    if cost_policy is None:
        cost_policy = CostPolicy.from_config()

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

    cost_result = expected_cost_action(risk_score, rul, cost_policy)

    return ServiceAlert(
        unit_id=unit_id,
        risk_score=round(float(risk_score), 4),
        estimated_rul_cycles=round(rul, 1),
        urgency=urgency,
        recommended_service_date=service_date.isoformat(),
        days_until_service=days,
        message=msg,
        action=cost_result["action"],
        expected_cost=round(cost_result["expected_cost"], 2),
    )


def sweep_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    rul: np.ndarray,
    thresholds: np.ndarray | None = None,
    cost_policy: CostPolicy | None = None,
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
    cost_policy : CostPolicy, optional
        When provided, adds an ``expected_cost`` column computed as the
        fleet-average expected cost at each threshold using
        ``expected_cost_action()``.

    Returns
    -------
    pd.DataFrame with columns:
        threshold, alerts, missed_failures, prevented_failures,
        false_service_calls, mean_lead_time_cycles, median_lead_time_cycles
        [, expected_cost  — only when cost_policy is given].
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

        row: dict = {
            "threshold": round(float(t), 3),
            "alerts": int(pred_pos.sum()),
            "missed_failures": int(fn_mask.sum()),
            "prevented_failures": int(tp_mask.sum()),
            "false_service_calls": int(fp_mask.sum()),
            "mean_lead_time_cycles": round(mean_lead, 2),
            "median_lead_time_cycles": round(median_lead, 2),
        }

        if cost_policy is not None:
            # Fleet-average expected cost: apply optimal action per unit at threshold t
            unit_costs = [
                expected_cost_action(score, r, cost_policy)["expected_cost"]
                for score, r in zip(y_score, rul)
            ]
            # At this threshold, flagged units are treated as "replace/inspect",
            # unflagged as "continue".  Use the per-unit optimal cost regardless
            # of threshold so the sweep shows the minimum achievable cost floor.
            row["expected_cost"] = round(float(np.mean(unit_costs)), 2)

        rows.append(row)

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
