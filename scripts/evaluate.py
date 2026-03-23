"""Evaluation script: make evaluate

Queries MLflow for all runs in pdm_cmapss and prints a comparison table
with Track A | Track B columns for every model and variant.

Usage:
    python -m scripts.evaluate [--experiment pdm_cmapss]
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="pdm_cmapss")
    return p.parse_args()


def main():
    args = _parse_args()

    import mlflow
    from mlops.tracking import setup_mlflow

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"Experiment '{args.experiment}' not found. Run 'make train' first.")
        return

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=500)
    if not runs:
        print("No runs found. Run 'make train' first.")
        return

    rows = []
    for run in runs:
        m = run.data.metrics
        p = run.data.params
        rows.append(
            {
                "model": run.info.run_name or p.get("model", "?"),
                "variant": p.get("variant", "cross"),
                "_start_time": run.info.start_time,
                # Track A
                "A_pr_auc": m.get("track_a_pr_auc", float("nan")),
                "A_recall@P80": m.get("track_a_recall_at_p80", float("nan")),
                "A_f1": m.get("track_a_f1", float("nan")),
                "A_brier": m.get("track_a_brier_score", float("nan")),
                "A_ece": m.get("track_a_ece", float("nan")),
                "A_c_index": m.get("track_a_c_index", float("nan")),
                # Track B
                "B_pr_auc": m.get("track_b_pr_auc", float("nan")),
                "B_recall@P80": m.get("track_b_recall_at_p80", float("nan")),
                "B_f1": m.get("track_b_f1", float("nan")),
                "B_brier": m.get("track_b_brier_score", float("nan")),
                "B_ece": m.get("track_b_ece", float("nan")),
                "B_c_index": m.get("track_b_c_index", float("nan")),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("_start_time", ascending=False)
        .drop_duplicates(subset=["model", "variant"])
        .drop(columns=["_start_time"])
        .sort_values(["model", "variant"])
    )

    # Separate deployment regressor runs from the classification/survival table
    is_deploy_reg = df["model"].str.contains("rul_regressor", case=False, na=False)
    df_clf = df[~is_deploy_reg].reset_index(drop=True)
    df_reg = df[is_deploy_reg].reset_index(drop=True)

    # Also fetch deploy_* metrics for regression rows directly from MLflow
    reg_rows = []
    for run in runs:
        if (run.info.run_name or "").startswith("xgboost_rul_regressor"):
            m = run.data.metrics
            reg_rows.append({
                "model": run.info.run_name,
                "variant": run.data.params.get("variant", "?"),
                "deploy_mae": m.get("deploy_mae", float("nan")),
                "deploy_rmse": m.get("deploy_rmse", float("nan")),
                "deploy_asymmetric": m.get("deploy_asymmetric_penalty", float("nan")),
                "_start_time": run.info.start_time,
            })
    if reg_rows:
        df_reg_detail = (
            pd.DataFrame(reg_rows)
            .sort_values("_start_time", ascending=False)
            .drop_duplicates(subset=["model", "variant"])
            .drop(columns=["_start_time"])
        )
    else:
        df_reg_detail = pd.DataFrame()

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 90)
    print("  Model Comparison — Track A (unit-holdout) | Track B (cross-domain)")
    print("=" * 90)
    print(df_clf.to_string(index=False))
    print("=" * 90 + "\n")

    if not df_reg_detail.empty:
        print("  Deployment Regressor (raw 24 features, FD001 test set)")
        print("  " + "-" * 50)
        print(df_reg_detail.to_string(index=False))
        print()

    # Best model per metric
    for col in ("A_pr_auc", "B_pr_auc"):
        best_row = df_clf.loc[df_clf[col].idxmax()]
        print(f"  Best {col}: {best_row['model']} ({best_row['variant']}) = {best_row[col]:.4f}")
    print()


if __name__ == "__main__":
    main()
