"""Unit tests for policy/notification.py — ServiceAlert + evaluate_alert()."""

from datetime import date

import pytest

from policy.notification import ServiceAlert, evaluate_alert


REF_DATE = date(2026, 1, 1)


# ---------------------------------------------------------------------------
# Urgency classification
# ---------------------------------------------------------------------------

def test_critical_on_high_risk():
    alert = evaluate_alert(1, risk_score=0.9, estimated_rul_cycles=50, reference_date=REF_DATE)
    assert alert.urgency == "critical"


def test_critical_on_low_rul_regardless_of_risk():
    alert = evaluate_alert(1, risk_score=0.2, estimated_rul_cycles=5, reference_date=REF_DATE)
    assert alert.urgency == "critical"


def test_critical_boundary_rul_exactly_10():
    alert = evaluate_alert(1, risk_score=0.1, estimated_rul_cycles=10, reference_date=REF_DATE)
    assert alert.urgency == "critical"


def test_critical_boundary_risk_exactly_08():
    alert = evaluate_alert(1, risk_score=0.8, estimated_rul_cycles=100, reference_date=REF_DATE)
    assert alert.urgency == "critical"


def test_high_urgency():
    alert = evaluate_alert(1, risk_score=0.6, estimated_rul_cycles=50, reference_date=REF_DATE)
    assert alert.urgency == "high"


def test_high_urgency_at_threshold():
    alert = evaluate_alert(1, risk_score=0.5, estimated_rul_cycles=50, reference_date=REF_DATE)
    assert alert.urgency == "high"


def test_scheduled_urgency():
    alert = evaluate_alert(1, risk_score=0.35, estimated_rul_cycles=80, reference_date=REF_DATE)
    assert alert.urgency == "scheduled"


def test_ok_urgency():
    alert = evaluate_alert(1, risk_score=0.1, estimated_rul_cycles=200, reference_date=REF_DATE)
    assert alert.urgency == "ok"


def test_custom_threshold_changes_high_boundary():
    alert = evaluate_alert(1, risk_score=0.6, estimated_rul_cycles=100, threshold=0.7, reference_date=REF_DATE)
    assert alert.urgency == "scheduled"


# ---------------------------------------------------------------------------
# Service date and days_until_service arithmetic
# ---------------------------------------------------------------------------

def test_days_until_service_computed_from_rul():
    # RUL = 30 cycles, 3 cycles/day → ceil(30/3) = 10 days
    alert = evaluate_alert(1, risk_score=0.1, estimated_rul_cycles=30,
                           cycles_per_day=3.0, reference_date=REF_DATE)
    assert alert.days_until_service == 10


def test_service_date_matches_days():
    alert = evaluate_alert(1, risk_score=0.1, estimated_rul_cycles=30,
                           cycles_per_day=3.0, reference_date=REF_DATE)
    expected = date(2026, 1, 11).isoformat()
    assert alert.recommended_service_date == expected


def test_zero_rul_gives_today_as_service_date():
    alert = evaluate_alert(1, risk_score=0.9, estimated_rul_cycles=0, reference_date=REF_DATE)
    assert alert.days_until_service == 0
    assert alert.recommended_service_date == REF_DATE.isoformat()


def test_negative_rul_clamped_to_zero():
    alert = evaluate_alert(1, risk_score=0.9, estimated_rul_cycles=-5, reference_date=REF_DATE)
    assert alert.days_until_service == 0


def test_fractional_rul_ceiled():
    # RUL = 1 cycle, 3 cycles/day → ceil(1/3) = 1 day
    alert = evaluate_alert(1, risk_score=0.1, estimated_rul_cycles=1,
                           cycles_per_day=3.0, reference_date=REF_DATE)
    assert alert.days_until_service == 1


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------

def test_returns_service_alert_instance():
    alert = evaluate_alert(1, risk_score=0.5, estimated_rul_cycles=60, reference_date=REF_DATE)
    assert isinstance(alert, ServiceAlert)


def test_unit_id_preserved():
    alert = evaluate_alert(42, risk_score=0.5, estimated_rul_cycles=60, reference_date=REF_DATE)
    assert alert.unit_id == 42


def test_message_contains_unit_id():
    alert = evaluate_alert(7, risk_score=0.5, estimated_rul_cycles=60, reference_date=REF_DATE)
    assert "7" in alert.message


def test_message_contains_service_date():
    alert = evaluate_alert(1, risk_score=0.5, estimated_rul_cycles=60,
                           cycles_per_day=3.0, reference_date=REF_DATE)
    assert alert.recommended_service_date in alert.message


def test_defaults_to_today_when_no_reference_date():
    alert = evaluate_alert(1, risk_score=0.1, estimated_rul_cycles=30)
    assert alert.recommended_service_date >= date.today().isoformat()
