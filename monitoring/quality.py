"""Real-time data quality checks for incoming sensor streams.

Checks:
  1. Flatline detection — rolling std < epsilon over a window of cycles
     (indicates sensor freeze / stuck value)
  2. Missing spike alert — >threshold fraction NaN in any rolling window
     (indicates dropout / transmission fault)

All alerts are returned as structured JSON-serialisable dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def check_flatline(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 10,
    epsilon: float = 1e-4,
) -> list[dict[str, Any]]:
    """Return alerts for sensors with rolling std < epsilon for `window` consecutive cycles."""
    alerts: list[dict[str, Any]] = []
    for col in sensor_cols:
        for unit_id, grp in df.groupby("unit_id"):
            grp = grp.sort_values("cycle")
            rolling_std = grp[col].rolling(window, min_periods=window // 2).std()
            flagged_cycles = grp["cycle"][rolling_std < epsilon].tolist()
            if flagged_cycles:
                alerts.append(
                    {
                        "type": "flatline",
                        "unit_id": int(unit_id),
                        "sensor": col,
                        "n_cycles": len(flagged_cycles),
                        "cycles": flagged_cycles[:20],  # truncate for readability
                    }
                )
    return alerts


def check_missing_spike(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 20,
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Return alerts when >threshold fraction of readings are NaN in any rolling window."""
    alerts: list[dict[str, Any]] = []
    for col in sensor_cols:
        for unit_id, grp in df.groupby("unit_id"):
            grp = grp.sort_values("cycle")
            nan_rate = grp[col].isna().rolling(window, min_periods=1).mean()
            spike_cycles = grp["cycle"][nan_rate > threshold].tolist()
            if spike_cycles:
                alerts.append(
                    {
                        "type": "missing_spike",
                        "unit_id": int(unit_id),
                        "sensor": col,
                        "max_nan_rate": round(float(nan_rate.max()), 4),
                        "n_cycles": len(spike_cycles),
                        "cycles": spike_cycles[:20],
                    }
                )
    return alerts


def run_quality_checks(
    df: pd.DataFrame,
    sensor_cols: list[str],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run all quality checks and optionally persist results as JSON.

    Returns a summary dict with flatline_alerts, missing_spike_alerts,
    and total_alerts count.
    """
    flatline = check_flatline(df, sensor_cols)
    missing = check_missing_spike(df, sensor_cols)

    result: dict[str, Any] = {
        "total_alerts": len(flatline) + len(missing),
        "flatline_alerts": flatline,
        "missing_spike_alerts": missing,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
