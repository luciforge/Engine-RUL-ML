"""Latency benchmark script: make benchmark

Loads the XGBoost deployment model (best_model.json) and the best research
model from MLflow, then reports p50/p95/p99 latency (ms) for both.

Usage:
    python -m scripts.benchmark [--n-runs 1000] [--batch-size 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-runs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--model-path",
        default=str(_PROJECT_ROOT / "artifacts" / "best_model.json"),
    )
    return p.parse_args()


def main():
    args = _parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Deployment model not found at {model_path}. Run 'make train' first.")
        return

    from evaluation.splits import track_a_split
    from data.loader import load_fd
    from labels.rul import add_rul
    from labels.binary import add_binary_label
    from service.benchmark import benchmark, print_report

    print("Loading data for benchmark…")
    ta = track_a_split("FD001")
    # Research model uses engineered features (276 cols)
    X_research = ta.test[ta.feature_cols].fillna(0.0).values.astype(np.float32)

    # Deployment model uses raw 24 features (3 op_settings + 21 sensors)
    _raw_sensor = [f"sensor_{i}" for i in range(1, 22)]
    _raw_op = [f"op_setting_{i}" for i in range(1, 4)]
    _raw_cols = _raw_op + _raw_sensor
    raw_test = add_binary_label(add_rul(load_fd("FD001", "test")))
    X_deploy = raw_test[_raw_cols].fillna(0.0).values.astype(np.float32)

    # Load the best research model from MLflow for comparison
    try:
        import yaml
        import mlflow
        from mlops.tracking import setup_mlflow

        setup_mlflow()
        client = mlflow.tracking.MlflowClient()
        with open(_PROJECT_ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        exp = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
        # In MLflow 3.x, LSTM training runs log only metrics; model artifacts are
        # stored as LoggedModel objects linked to their source run via source_run_id.
        # Retrieve the LoggedModel whose source run recorded the highest track_a_pr_auc.
        all_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.track_a_pr_auc > 0",
            max_results=500,
        )
        run_auc = {
            r.info.run_id: r.data.metrics.get("track_a_pr_auc", 0.0)
            for r in all_runs
        }
        logged = client.search_logged_models(
            experiment_ids=[exp.experiment_id],
            max_results=200,
        )
        # Keep only models whose source run has a recorded AUC
        candidates = [m for m in logged if m.source_run_id in run_auc]
        if not candidates:
            raise ValueError("No logged models with track_a_pr_auc found")
        best_model = max(candidates, key=lambda m: run_auc[m.source_run_id])
        best_run_name = next(
            r.info.run_name for r in all_runs
            if r.info.run_id == best_model.source_run_id
        )
        model_uri = f"models:/{best_model.model_id}"
        research_model = mlflow.sklearn.load_model(model_uri)
        research_fn = lambda arr: research_model.predict_proba(arr)  # noqa: E731
        research_label = f"{best_run_name} (MLflow, AUC={run_auc[best_model.source_run_id]:.4f})"
    except Exception as exc:
        print(f"Could not load research model from MLflow ({exc}). Using dummy baseline.")
        from sklearn.dummy import DummyClassifier
        dc = DummyClassifier(strategy="uniform")
        dc.fit(X_research[:10], [0, 1] * 5)
        research_fn = lambda arr: dc.predict_proba(arr)  # noqa: E731
        research_label = "DummyClassifier (fallback)"

    from xgboost import XGBClassifier
    deploy_clf = XGBClassifier()
    deploy_clf.load_model(str(model_path))
    deploy_fn = lambda arr: deploy_clf.predict_proba(arr)  # noqa: E731

    print(f"\nRunning benchmark: {args.n_runs} runs, batch_size={args.batch_size}")
    print(f"Research model  : {research_label}  ({X_research.shape[1]} features)")
    print(f"Deployment model: {model_path.name}  ({X_deploy.shape[1]} features)")
    result = benchmark(
        research_fn=research_fn,
        deploy_fn=deploy_fn,
        X_research=X_research,
        X_deploy=X_deploy,
        warmup=1000,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
    )
    print_report(result)


if __name__ == "__main__":
    main()
