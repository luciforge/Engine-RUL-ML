"""Tests: FastAPI /health and /predict endpoints.

Uses a mock ONNX session so no real model file is required.
Validates response types and that risk_score ∈ [0, 1].
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient


class _MockClf:
    """Minimal stand-in for XGBClassifier that returns fixed probabilities."""

    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 2), dtype=np.float64)
        probs[:, 0] = 0.7
        probs[:, 1] = 0.3
        return probs


class _MockReg:
    """Minimal stand-in for XGBRegressor that returns fixed RUL."""

    def predict(self, X):
        return np.full(len(X), 100.0, dtype=np.float64)


@pytest.fixture
def client(monkeypatch):
    import service.api as api_mod

    monkeypatch.setattr(api_mod, "_clf_model", _MockClf())
    monkeypatch.setattr(api_mod, "_reg_model", _MockReg())
    return TestClient(api_mod.app)


def _sensor_payload() -> dict:
    return {
        "unit_id": 1,
        "cycle": 50,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
        # Mid-range values within schema bounds for each sensor
        "sensor_1": 518.0,   # unconstrained fan inlet temp
        "sensor_2": 600.0,   # [500, 700] LPC outlet temp
        "sensor_3": 1450.0,  # [1200, 1700] HPC outlet temp
        "sensor_4": 1350.0,  # [1000, 1700] LPT outlet temp
        "sensor_5": 9.0,     # unconstrained fan inlet pressure
        "sensor_6": 21.0,    # unconstrained bypass-duct pressure
        "sensor_7": 550.0,   # [500, 600] HPC outlet pressure
        "sensor_8": 2388.0,  # [2300, 2500] physical fan speed
        "sensor_9": 8900.0,  # [7000, 10000] physical core speed
        "sensor_10": 1.3,    # unconstrained engine pressure ratio
        "sensor_11": 48.0,   # [38, 60] HPC outlet static pressure
        "sensor_12": 530.0,  # [500, 560] fuel flow ratio
        "sensor_13": 2388.0, # [2300, 2500] corrected fan speed
        "sensor_14": 8900.0, # [7000, 10000] corrected core speed
        "sensor_15": 8.4,    # unconstrained bypass ratio
        "sensor_16": 0.03,   # unconstrained burner fuel-air ratio
        "sensor_17": 390.0,  # unconstrained bleed enthalpy
        "sensor_18": 2300.0, # unconstrained required fan speed
        "sensor_19": 100.0,  # [80, 110] required fan conversion speed
        "sensor_20": 38.0,   # [20, 50] HPT coolant bleed
        "sensor_21": 23.0,   # unconstrained LPT coolant bleed
    }



def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_payload(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "rul_model_loaded" in body
    assert "rul_model_path" in body



def test_predict_returns_200(client):
    r = client.post("/predict", json=_sensor_payload())
    assert r.status_code == 200


def test_predict_risk_score_in_range(client):
    body = client.post("/predict", json=_sensor_payload()).json()
    assert 0.0 <= body["risk_score"] <= 1.0


def test_predict_response_typed(client):
    body = client.post("/predict", json=_sensor_payload()).json()
    assert isinstance(body["risk_score"], float)
    assert isinstance(body["replace_within_30"], bool)
    assert isinstance(body["estimated_rul"], float)
    assert body["unit_id"] == 1


def test_predict_replace_within_30_consistent(client):
    body = client.post("/predict", json=_sensor_payload()).json()
    # With mock returning 0.3 risk, threshold=0.5 → should be False
    assert body["replace_within_30"] == (body["risk_score"] >= 0.5)


def test_predict_missing_field_returns_422(client):
    payload = _sensor_payload()
    del payload["op_setting_1"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_op_setting_3_out_of_range_returns_422(client):
    payload = _sensor_payload()
    payload["op_setting_3"] = 999.0
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
