"""Pydantic v2 request/response schemas for the predictive maintenance API.

Sensor value ranges are validated against the CMAPSS dataset distribution.
op_setting_3 encodes altitude (0 = Sea Level, 100 = cruise) across all variants.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    """Single-engine snapshot for /predict.

    All 26 fields from the CMAPSS column layout are required.
    """

    unit_id: int = Field(..., ge=1, description="Engine unit identifier")
    cycle: int = Field(..., ge=1, description="Current operational cycle")

    # Operational settings
    op_setting_1: float = Field(..., ge=-1.0, le=1.0, description="Operational setting 1")
    op_setting_2: float = Field(..., ge=-1.0, le=1.0, description="Operational setting 2")
    op_setting_3: float = Field(..., ge=0.0, le=100.0, description="Operational setting 3 (altitude)")

    # Sensors — ranges validated against CMAPSS min/max (+10% margin)
    sensor_1: float  # Fan inlet temperature (~518)
    sensor_2: float = Field(..., ge=500.0, le=700.0)   # LPC outlet temperature
    sensor_3: float = Field(..., ge=1200.0, le=1700.0)  # HPC outlet temperature
    sensor_4: float = Field(..., ge=1000.0, le=1700.0)  # LPT outlet temperature
    sensor_5: float  # Fan inlet pressure
    sensor_6: float  # Bypass-duct pressure
    sensor_7: float = Field(..., ge=500.0, le=600.0)   # HPC outlet pressure
    sensor_8: float = Field(..., ge=2300.0, le=2500.0)  # Physical fan speed
    sensor_9: float = Field(..., ge=7000.0, le=10000.0) # Physical core speed
    sensor_10: float  # Engine pressure ratio
    sensor_11: float = Field(..., ge=38.0, le=60.0)    # HPC outlet static pressure
    sensor_12: float = Field(..., ge=500.0, le=560.0)  # Fuel flow ratio
    sensor_13: float = Field(..., ge=2300.0, le=2500.0) # Corrected fan speed
    sensor_14: float = Field(..., ge=7000.0, le=10000.0) # Corrected core speed
    sensor_15: float  # Bypass ratio
    sensor_16: float  # Burner fuel-air ratio
    sensor_17: float  # Bleed enthalpy
    sensor_18: float  # Required fan speed
    sensor_19: float = Field(..., ge=80.0, le=110.0)   # Required fan conversion speed
    sensor_20: float = Field(..., ge=20.0, le=50.0)    # HPT coolant bleed
    sensor_21: float  # LPT coolant bleed


class PredictResponse(BaseModel):
    unit_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0, description="P(failure within X cycles)")
    replace_within_30: bool = Field(..., description="True when risk_score >= threshold")
    estimated_rul: float = Field(..., ge=0.0, description="Estimated remaining useful life (cycles)")
    # Uncertainty interval fields — populated when conformal artifact is available
    risk_lower: float | None = Field(None, ge=0.0, le=1.0, description="Conformal lower bound on risk score")
    risk_upper: float | None = Field(None, ge=0.0, le=1.0, description="Conformal upper bound on risk score")
    rul_lower: float | None = Field(None, ge=0.0, description="Lower quantile RUL estimate")
    rul_upper: float | None = Field(None, ge=0.0, description="Upper quantile RUL estimate")


class BatchScoreResponse(BaseModel):
    rows_scored: int
    output_path: str
    message: str = "Scored parquet written to output_path"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    rul_model_loaded: bool = False
    rul_model_path: str = ""


class ExplainResponse(BaseModel):
    unit_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0, description="P(failure within X cycles)")
    shap_values: dict[str, float] = Field(
        ...,
        description=(
            "Top-N feature SHAP contributions (log-odds space). "
            "Positive → pushes toward failure; negative → pushes away."
        ),
    )
    base_value: float = Field(
        ...,
        description="Model baseline expected value (log-odds). base_value + sum(shap_values) = raw model output.",
    )
    top_n: int = Field(default=10, description="Number of features returned, ranked by |SHAP value|.")


class ScheduleResponse(BaseModel):
    unit_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    estimated_rul_cycles: float = Field(..., ge=0.0)
    urgency: str = Field(..., description="critical | high | scheduled | ok")
    recommended_service_date: str = Field(..., description="ISO-8601 date")
    days_until_service: int
    message: str = ""
    action: str = Field("", description="Optimal cost-policy action: replace | inspect | continue")
    expected_cost: float = Field(0.0, description="Expected cost (EUR) of the recommended action")
    message: str
