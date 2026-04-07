"""RandomForest + XGBoost classifiers with Optuna HPO.

HPO runs on the Track A training split (XGBoost only).
A 10% internal validation slice is held out from training for HPO scoring.
RandomForest uses sensible fixed hyperparameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from features.pipeline import get_feature_cols

optuna.logging.set_verbosity(optuna.logging.WARNING)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _Xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df["label_within_x"].values.astype(int)
    return X, y


def train_random_forest(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> RandomForestClassifier:
    """Train RandomForestClassifier with fixed hyperparameters."""
    cfg = _cfg()
    X, y = _Xy(train, feature_cols)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg["training"]["random_seed"],
    )
    rf.fit(X, y)
    return rf


def _xgb_objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_score = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, y_score))


def train_xgboost(
    train: pd.DataFrame,
    feature_cols: list[str],
    run_hpo: bool = True,
) -> XGBClassifier:
    """Train XGBoost with optional Optuna HPO (50 trials, maximise PR-AUC).

    If run_hpo=True, a 10% internal validation slice is used for HPO scoring.
    """
    cfg = _cfg()
    X, y = _Xy(train, feature_cols)

    if run_hpo:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=cfg["training"]["random_seed"], stratify=y
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _xgb_objective(trial, X_tr, y_tr, X_val, y_val),
            n_trials=cfg["optuna"]["trials"],
            show_progress_bar=True,
        )
        best_params = study.best_params
        best_params.update(
            {"eval_metric": "logloss", "tree_method": "hist", "random_state": 42}
        )
        model = XGBClassifier(**best_params)
    else:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=cfg["training"]["random_seed"],
        )

    model.fit(X, y)
    return model


def calibrate_xgboost(
    clf: XGBClassifier,
    calib_df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float | None = None,
) -> dict:
    """Compute a split-conformal nonconformity threshold for the classifier.

    Uses the calibration set to find ``q_hat``: the (1-alpha) quantile of
    nonconformity scores ``1 - p_true_class``.  At inference, a prediction
    set is then:  ``risk >= threshold - q_hat``  (single-label binary case).

    In practice the result is used to widen the point estimate into an
    interval:  risk_lower = clip(risk - q_hat, 0, 1),
               risk_upper = clip(risk + q_hat, 0, 1).

    Parameters
    ----------
    clf          : fitted XGBClassifier
    calib_df     : held-out calibration DataFrame (never seen during training)
    feature_cols : feature column names used during training
    alpha        : miscoverage rate (default from config: calibration.conformal_alpha)

    Returns
    -------
    dict with keys ``alpha`` (float) and ``q_hat`` (float)
    """
    cfg = _cfg()
    if alpha is None:
        alpha = float(cfg.get("calibration", {}).get("conformal_alpha", 0.1))

    X_cal, y_cal = _Xy(calib_df, feature_cols)
    proba = clf.predict_proba(X_cal)  # shape (N, 2)
    # Nonconformity score: 1 - probability assigned to the true class
    scores = 1.0 - proba[np.arange(len(y_cal)), y_cal]

    n = len(scores)
    # Finite-sample corrected quantile level
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    q_hat = float(np.quantile(scores, level))

    return {"alpha": alpha, "q_hat": q_hat}


def save_conformal_artifact(q_hat_dict: dict, path: Path) -> Path:
    """Persist the conformal calibration artifact as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(q_hat_dict, f, indent=2)
    return path


def load_conformal_artifact(path: Path) -> dict | None:
    """Load a conformal calibration artifact.  Returns None if file absent."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def train_xgboost_quantile(
    train: pd.DataFrame,
    feature_cols: list[str],
    quantiles: tuple[float, float] = (0.1, 0.9),
) -> tuple[XGBRegressor, XGBRegressor]:
    """Train lower and upper quantile XGBoost regressors for RUL intervals.

    Uses ``objective="reg:quantileerror"`` (available in XGBoost >= 2.0).
    The label column is ``rul`` which must be present in ``train``.

    Parameters
    ----------
    train        : training DataFrame containing 'rul' and feature_cols
    feature_cols : feature column names
    quantiles    : (lower_q, upper_q) quantile pair, default (0.1, 0.9)

    Returns
    -------
    (lower_regressor, upper_regressor) — both XGBRegressor instances
    """
    cfg = _cfg()
    seed = cfg["training"]["random_seed"]
    qlow, qhigh = quantiles

    X = train[feature_cols].fillna(0.0).values.astype(np.float32)
    y = train["rul"].values.astype(np.float32)

    base_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
        objective="reg:quantileerror",
    )

    lower = XGBRegressor(**base_params, quantile_alpha=qlow)
    upper = XGBRegressor(**base_params, quantile_alpha=qhigh)

    lower.fit(X, y)
    upper.fit(X, y)

    return lower, upper
