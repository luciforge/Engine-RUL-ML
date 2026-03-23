"""Drift monitoring script: make drift-report

Generates an evidently DataDriftPreset report comparing the training
feature distribution (reference) against the test feature window (current).

Usage:
    python -m scripts.drift [--variant FD001] [--output-dir reports/drift]
"""

from __future__ import annotations

import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="FD001")
    p.add_argument("--output-dir", default=str(_PROJECT_ROOT / "reports" / "drift"))
    return p.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)

    from evaluation.splits import track_a_split
    from monitoring.drift import run_drift_report
    from monitoring.quality import run_quality_checks, check_flatline

    print(f"Building Track A split for {args.variant}…")
    ta = track_a_split(args.variant)

    print("Running evidently drift report…")
    summary = run_drift_report(
        reference=ta.train,
        current=ta.test,
        feature_cols=ta.feature_cols,
        output_dir=output_dir,
    )

    print("\nDrift summary:")
    for k, v in summary.items():
        if k != "drifted_columns":
            print(f"  {k}: {v}")
    if summary["drifted_columns"]:
        print(f"  drifted_columns (top 10): {summary['drifted_columns'][:10]}")

    # Data quality checks on test window
    print("\nRunning data quality checks…")
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    quality = run_quality_checks(
        df=ta.test,
        sensor_cols=sensor_cols,
        output_path=output_dir / "quality_alerts.json",
    )
    print(f"  Total quality alerts: {quality['total_alerts']}")
    print(f"  Flatline alerts: {len(quality['flatline_alerts'])}")
    print(f"  Missing-spike alerts: {len(quality['missing_spike_alerts'])}")


if __name__ == "__main__":
    main()
