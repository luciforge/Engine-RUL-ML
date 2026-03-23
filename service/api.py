"""FastAPI predictive maintenance inference service.

Endpoints:
    GET  /health          — liveness check; reports model load status
    POST /predict         — single-engine risk score + estimated RUL
    POST /explain         — XGBoost native SHAP attributions for a prediction
    POST /schedule        — maintenance recommendation with urgency + service date
    POST /batch_score     — upload feature CSV, download scored parquet

Models are XGBoost JSON format (24 raw features: 3 op_settings + 21 sensors).
Override paths with env vars ONNX_MODEL_PATH / ONNX_RUL_MODEL_PATH.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier as _XGBClf, XGBRegressor as _XGBReg
from fastapi import FastAPI, File, HTTPException, UploadFile

from service.schemas import BatchScoreResponse, ExplainResponse, HealthResponse, PredictResponse, ScheduleResponse, SensorInput

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
_OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "best_model.json"
_DEFAULT_RUL_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "rul_regressor.json"
_THRESHOLD = 0.5

_clf_model: _XGBClf | None = None
_reg_model: _XGBReg | None = None


def _cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _get_model_path() -> Path:
    env = os.environ.get("ONNX_MODEL_PATH")
    return Path(env) if env else _DEFAULT_MODEL_PATH


def _get_rul_model_path() -> Path:
    env = os.environ.get("ONNX_RUL_MODEL_PATH")
    return Path(env) if env else _DEFAULT_RUL_MODEL_PATH


def _load_clf() -> _XGBClf:
    global _clf_model
    if _clf_model is None:
        _model_path = _get_model_path()
        if not _model_path.exists():
            raise RuntimeError(f"Classifier model not found at {_model_path}")
        _clf_model = _XGBClf()
        _clf_model.load_model(str(_model_path))
    return _clf_model


def _load_reg() -> _XGBReg | None:
    """Load the RUL regressor. Returns None if the model file is absent
    (graceful degradation — fallback kicks in at inference time)."""
    global _reg_model
    if _reg_model is None:
        rul_path = _get_rul_model_path()
        if rul_path.exists():
            _reg_model = _XGBReg()
            _reg_model.load_model(str(rul_path))
    return _reg_model


def _df_to_array(df: pd.DataFrame) -> np.ndarray:
    return df[_OP_COLS + _SENSOR_COLS].fillna(0.0).values.astype(np.float32)


def _infer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run inference using XGBoost deployment models. Returns (risk_scores, estimated_rul).

    risk_scores   — positive-class probability from best_model.json
    estimated_rul — predicted cycles from rul_regressor.json; falls back to a
                    config-driven linear proxy if the regressor file is absent.
    """
    clf = _load_clf()
    risk_scores = clf.predict_proba(arr)[:, 1].astype(float)

    reg = _load_reg()
    if reg is not None:
        estimated_rul = np.clip(reg.predict(arr).astype(float), 0.0, None)
    else:
        # Fallback: linear inverse of risk score scaled by configured max RUL.
        # Activated only when rul_regressor.json has not been built yet.
        max_rul = float(_cfg()["service"].get("max_rul_cycles", 340))
        estimated_rul = (1.0 - risk_scores) * max_rul

    return risk_scores, estimated_rul


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        _load_clf()
        loaded = True
    except RuntimeError:
        loaded = False
    rul_loaded = _load_reg() is not None
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_path=str(_get_model_path()),
        rul_model_loaded=rul_loaded,
        rul_model_path=str(_get_rul_model_path()),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: SensorInput):
    df = pd.DataFrame([payload.model_dump()])
    arr = _df_to_array(df)
    try:
        risk_scores, rul_estimates = _infer(arr)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    risk = float(np.clip(risk_scores[0], 0.0, 1.0))
    rul = float(max(0.0, rul_estimates[0]))
    return PredictResponse(
        unit_id=payload.unit_id,
        risk_score=risk,
        replace_within_30=risk >= _THRESHOLD,
        estimated_rul=rul,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: SensorInput, top_n: int = 10):
    from monitoring.explainability import shap_explain

    df = pd.DataFrame([payload.model_dump()])
    arr = _df_to_array(df)
    try:
        risk_scores, _ = _infer(arr)
        attribution, base_value = shap_explain(_load_clf(), arr, top_n=top_n)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ExplainResponse(
        unit_id=payload.unit_id,
        risk_score=float(np.clip(risk_scores[0], 0.0, 1.0)),
        shap_values=attribution,
        base_value=base_value,
        top_n=top_n,
    )


@app.post("/schedule", response_model=ScheduleResponse)
def schedule(payload: SensorInput, cycles_per_day: float = 3.0):
    """Return a maintenance schedule recommendation for a single engine.

    Combines the risk classifier and RUL regressor outputs with
    policy.notification.evaluate_alert() to produce a structured alert:
    urgency level (critical / high / scheduled / ok) and a recommended
    service date based on estimated RUL and assumed operational rate.
    """
    from policy.notification import evaluate_alert

    df = pd.DataFrame([payload.model_dump()])
    arr = _df_to_array(df)
    try:
        risk_scores, rul_estimates = _infer(arr)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    alert = evaluate_alert(
        unit_id=payload.unit_id,
        risk_score=float(np.clip(risk_scores[0], 0.0, 1.0)),
        estimated_rul_cycles=float(max(0.0, rul_estimates[0])),
        threshold=_THRESHOLD,
        cycles_per_day=cycles_per_day,
    )
    return ScheduleResponse(
        unit_id=alert.unit_id,
        risk_score=alert.risk_score,
        estimated_rul_cycles=alert.estimated_rul_cycles,
        urgency=alert.urgency,
        recommended_service_date=alert.recommended_service_date,
        days_until_service=alert.days_until_service,
        message=alert.message,
    )


@app.post("/batch_score", response_model=BatchScoreResponse)
async def batch_score(file: UploadFile = File(...)):
    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {exc}") from exc

    required = set(_OP_COLS + _SENSOR_COLS)
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    arr = _df_to_array(df)
    try:
        risk_scores, rul_estimates = _infer(arr)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df["risk_score"] = np.clip(risk_scores, 0.0, 1.0)
    df["replace_within_30"] = risk_scores >= _THRESHOLD
    df["estimated_rul"] = np.clip(rul_estimates, 0.0, None)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        out_path = tmp.name
    df.to_parquet(out_path, index=False)

    return BatchScoreResponse(rows_scored=len(df), output_path=out_path)
