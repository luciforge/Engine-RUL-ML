"""MLflow experiment tracking helpers.

By default uses the URI in config.yaml (Postgres for Docker Compose).
Override with env var MLFLOW_TRACKING_URI for local dev:
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db make train
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def setup_mlflow() -> str:
    """Configure MLflow tracking URI and experiment. Returns experiment_id."""
    cfg = _cfg()

    # Env var takes priority (local dev / CI)
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"].get(
        "local_tracking_uri", "sqlite:///mlflow.db"
    )
    mlflow.set_tracking_uri(uri)

    experiment_name = cfg["mlflow"]["experiment_name"]
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


def log_run(
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, Path] | None = None,
    model_obj=None,
    feature_importance: dict[str, float] | None = None,
) -> str:
    """Log a complete training run and return the run_id.

    Parameters
    ----------
    model_name : str         Run name and model identifier.
    params : dict            Hyperparameters to log.
    metrics : dict           All metrics (Track A + Track B) to log.
    artifacts : dict         name → Path for additional artifact files.
    model_obj                sklearn-compatible model (logged via mlflow.sklearn).
    feature_importance : dict  Feature name → importance score for bar plot.
    """
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(str(path), artifact_path=name)

        if feature_importance:
            import matplotlib.pyplot as plt
            import pandas as pd

            fi = (
                pd.Series(feature_importance)
                .sort_values(ascending=False)
                .head(30)
            )
            fig, ax = plt.subplots(figsize=(8, 7))
            fi.plot.barh(ax=ax)
            ax.set_title(f"Top-30 features — {model_name}")
            ax.invert_yaxis()
            plt.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix=f"{model_name}_fi_"
            ) as tmp:
                tmp_path = tmp.name
            fig.savefig(tmp_path, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(tmp_path, artifact_path="plots")

        if model_obj is not None:
            try:
                mlflow.sklearn.log_model(model_obj, name="model")
            except Exception:
                pass  # Non-sklearn models (e.g. PyTorch) handled separately

        return run.info.run_id
