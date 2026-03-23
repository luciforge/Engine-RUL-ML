"""Feature drift detection using evidently DataDriftPreset.

Reference distribution: training feature snapshot.
Current distribution:   rolling test window (configurable size).

Outputs:
  - drift_report.html    — full evidently HTML report
  - drift_summary.json   — machine-readable drift summary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Generate an evidently DataDriftPreset report comparing reference vs current features.

    Writes drift_report.html and drift_summary.json to output_dir.
    Returns a dict with drift summary statistics.
    """
    from evidently.core.report import Report
    from evidently.presets import DataDriftPreset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref = reference[feature_cols].copy()
    cur = current[feature_cols].copy()

    report = Report([DataDriftPreset()])
    # New API: run(current, reference) returns a Snapshot
    with np.errstate(invalid="ignore"):
        snapshot = report.run(cur, ref)

    html_path = output_dir / "drift_report.html"
    snapshot.save_html(str(html_path))

    raw = snapshot.dict()

    # Extract per-feature drift flags from the evidently result structure
    drifted: list[str] = []
    for m in raw.get("metrics", []):
        result = m.get("result", {})
        if isinstance(result, dict) and result.get("drift_detected", False):
            col = result.get("column_name") or result.get("feature_name")
            if col:
                drifted.append(str(col))

    summary: dict[str, Any] = {
        "total_features": len(feature_cols),
        "drifted_features": len(drifted),
        "drift_share": round(len(drifted) / max(len(feature_cols), 1), 4),
        "drifted_columns": sorted(drifted),
        "report_path": str(html_path),
    }

    json_path = output_dir / "drift_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Drift report saved to {html_path}\n"
        f"  {len(drifted)}/{len(feature_cols)} features drifted "
        f"({summary['drift_share']:.1%})"
    )
    return summary
