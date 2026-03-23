"""Training orchestrator: make train

Trains all models on Track A (per-variant unit-holdout) and Track B (cross-domain).
Logs every run to MLflow. Exports the best XGBoost deployment model to JSON.

Usage:
    python -m scripts.train [--no-hpo] [--no-lstm] [--variants FD001 FD002 ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-hpo", action="store_true", help="Skip Optuna HPO (faster dev runs)")
    p.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    p.add_argument(
        "--variants",
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="FD variants to run Track A on",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    from mlops.tracking import setup_mlflow, log_run
    from evaluation.splits import track_a_split, track_b_split
    from evaluation.metrics import classification_metrics, regression_metrics
    from models.baseline.threshold import ThresholdClassifier
    from models.baseline.logistic import LogisticBaseline
    from models.classical.rf_xgb import train_random_forest, train_xgboost
    from models.survival.cox import train_cox, evaluate_c_index
    from service.onnx_export import export_xgboost

    setup_mlflow()

    log.info("Building Track B split…")
    tb = track_b_split()
    best_xgb_prauc = -1.0
    best_xgb_model = None
    best_xgb_feature_cols = None

    for variant in args.variants:
        log.info("=== Track A — %s ===", variant)
        ta = track_a_split(variant)

        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

        thresh = ThresholdClassifier().fit(ta.train, sensor_cols)
        ta_scores_thresh = thresh.predict_proba(ta.test)
        ta_m_thresh = classification_metrics(ta.test["label_within_x"].values, ta_scores_thresh)
        tb_scores_thresh = thresh.predict_proba(tb.test)
        tb_m_thresh = classification_metrics(tb.test["label_within_x"].values, tb_scores_thresh)

        log_run(
            model_name=f"threshold_{variant}",
            params={"model": "threshold", "variant": variant, "sensor": thresh.sensor_col_},
            metrics={
                **{f"track_a_{k}": v for k, v in ta_m_thresh.items()},
                **{f"track_b_{k}": v for k, v in tb_m_thresh.items()},
            },
        )

        logistic = LogisticBaseline().fit(ta.train)
        ta_m_log = classification_metrics(
            ta.test["label_within_x"].values, logistic.predict_proba(ta.test)
        )
        tb_m_log = classification_metrics(
            tb.test["label_within_x"].values, logistic.predict_proba(tb.test)
        )
        log_run(
            model_name=f"logistic_{variant}",
            params={"model": "logistic_regression", "variant": variant},
            metrics={
                **{f"track_a_{k}": v for k, v in ta_m_log.items()},
                **{f"track_b_{k}": v for k, v in tb_m_log.items()},
            },
            model_obj=logistic.sklearn_model,
        )

        rf = train_random_forest(ta.train, ta.feature_cols)
        X_ta_test = ta.test[ta.feature_cols].fillna(0).values
        ta_scores_rf = rf.predict_proba(X_ta_test)[:, 1]
        ta_m_rf = classification_metrics(ta.test["label_within_x"].values, ta_scores_rf)
        fi_rf = dict(zip(ta.feature_cols, rf.feature_importances_))
        # Track B (RF trained on Track A, evaluated on Track B)
        tb_shared = [c for c in ta.feature_cols if c in tb.test.columns]
        tb_scores_rf = rf.predict_proba(tb.test[tb_shared].fillna(0).values)[:, 1]
        tb_m_rf = classification_metrics(tb.test["label_within_x"].values, tb_scores_rf)
        log_run(
            model_name=f"random_forest_{variant}",
            params={"model": "random_forest", "variant": variant, "n_estimators": 200},
            metrics={
                **{f"track_a_{k}": v for k, v in ta_m_rf.items()},
                **{f"track_b_{k}": v for k, v in tb_m_rf.items()},
            },
            model_obj=rf,
            feature_importance=fi_rf,
        )

        log.info("Training XGBoost (HPO=%s)…", not args.no_hpo)
        xgb = train_xgboost(ta.train, ta.feature_cols, run_hpo=not args.no_hpo)
        ta_scores_xgb = xgb.predict_proba(X_ta_test)[:, 1]
        ta_m_xgb = classification_metrics(ta.test["label_within_x"].values, ta_scores_xgb)
        fi_xgb = dict(zip(ta.feature_cols, xgb.feature_importances_))
        tb_scores_xgb = xgb.predict_proba(tb.test[tb_shared].fillna(0).values)[:, 1]
        tb_m_xgb = classification_metrics(tb.test["label_within_x"].values, tb_scores_xgb)
        log_run(
            model_name=f"xgboost_{variant}",
            params={"model": "xgboost", "variant": variant},
            metrics={
                **{f"track_a_{k}": v for k, v in ta_m_xgb.items()},
                **{f"track_b_{k}": v for k, v in tb_m_xgb.items()},
            },
            model_obj=xgb,
            feature_importance=fi_xgb,
        )

        if ta_m_xgb["pr_auc"] > best_xgb_prauc:
            best_xgb_prauc = ta_m_xgb["pr_auc"]
            best_xgb_model = xgb
            best_xgb_feature_cols = ta.feature_cols

        log.info("Training CoxPH…")
        try:
            cph = train_cox(ta.train, ta.feature_cols)
            from models.survival.cox import evaluate_c_index
            ci_ta = evaluate_c_index(cph, ta.test, ta.feature_cols)
            ci_tb = evaluate_c_index(cph, tb.test, ta.feature_cols)
            log_run(
                model_name=f"coxph_{variant}",
                params={"model": "coxph", "variant": variant, "penalizer": 0.1},
                metrics={"track_a_c_index": ci_ta, "track_b_c_index": ci_tb},
            )
        except Exception as exc:
            log.warning("CoxPH failed for %s: %s", variant, exc)

        if not args.no_lstm:
            log.info("Training LSTM…")
            try:
                from models.deep.lstm import train_lstm, predict_proba_lstm
                import yaml
                with open(_PROJECT_ROOT / "config.yaml") as f:
                    cfg = yaml.safe_load(f)
                # 10% validation slice from train split
                unit_ids = ta.train["unit_id"].unique()
                n_val = max(1, int(len(unit_ids) * 0.1))
                val_ids = set(unit_ids[:n_val])
                lstm_train = ta.train[~ta.train["unit_id"].isin(val_ids)]
                lstm_val = ta.train[ta.train["unit_id"].isin(val_ids)]

                lstm = train_lstm(lstm_train, lstm_val, ta.feature_cols)
                window = cfg["features"]["lstm_window"]
                ta_scores_lstm = predict_proba_lstm(lstm, ta.test, ta.feature_cols, window)
                from models.deep.lstm import SlidingWindowDataset
                ds = SlidingWindowDataset(ta.test, ta.feature_cols, window)
                ta_labels_lstm = np.array([ds[i][1].item() for i in range(len(ds))])

                if len(ta_scores_lstm) > 0 and len(ta_labels_lstm) > 0:
                    ta_m_lstm = classification_metrics(ta_labels_lstm, ta_scores_lstm)
                    log_run(
                        model_name=f"lstm_{variant}",
                        params={"model": "lstm", "variant": variant, "window": window},
                        metrics={f"track_a_{k}": v for k, v in ta_m_lstm.items()},
                    )
            except Exception as exc:
                log.warning("LSTM failed for %s: %s", variant, exc)

    # Deployment-track models use the raw 24 input features (3 operational settings +
    # 21 sensors) to support single-cycle inference without pre-computed rolling windows.
    log.info("Training deployment models on raw features…")
    from xgboost import XGBRegressor
    from evaluation.metrics import regression_metrics
    from data.loader import load_fd as _load_raw
    from labels.rul import add_rul as _add_rul
    from labels.binary import add_binary_label as _add_bin

    _raw_sensor = [f"sensor_{i}" for i in range(1, 22)]
    _raw_op = [f"op_setting_{i}" for i in range(1, 4)]
    _raw_cols = _raw_op + _raw_sensor

    raw_train = _add_bin(_add_rul(_load_raw("FD001", "train")))
    raw_test = _add_bin(_add_rul(_load_raw("FD001", "test")))
    X_raw_tr = raw_train[_raw_cols].fillna(0.0).values.astype(np.float32)
    y_cls_raw = raw_train["label_within_x"].values
    y_rul_raw = raw_train["rul"].values.astype(np.float32)

    from xgboost import XGBClassifier as _XGBClf
    deploy_clf = _XGBClf(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3.0,
        eval_metric="logloss", tree_method="hist", random_state=42,
    )
    deploy_clf.fit(X_raw_tr, y_cls_raw)
    clf_json = _PROJECT_ROOT / "artifacts" / "best_model.json"
    saved_clf = export_xgboost(deploy_clf, _raw_cols, clf_json)
    log.info("Deployment classifier exported to %s", saved_clf)

    deploy_reg = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42,
    )
    deploy_reg.fit(X_raw_tr, y_rul_raw)
    reg_json = _PROJECT_ROOT / "artifacts" / "rul_regressor.json"
    saved_reg = export_xgboost(deploy_reg, _raw_cols, reg_json)
    log.info("Deployment RUL regressor exported to %s", saved_reg)

    X_raw_te = raw_test[_raw_cols].fillna(0.0).values.astype(np.float32)
    y_rul_te = raw_test["rul"].values.astype(np.float32)
    rul_pred = deploy_reg.predict(X_raw_te)
    reg_m = regression_metrics(y_rul_te, rul_pred)
    log_run(
        model_name="xgboost_rul_regressor_deploy",
        params={"model": "xgboost_regressor", "variant": "FD001", "features": "raw"},
        metrics={f"deploy_{k}": v for k, v in reg_m.items()},
    )
    log.info(
        "Deployment regressor — MAE=%.2f  RMSE=%.2f  Asymmetric=%.2f",
        reg_m["mae"], reg_m["rmse"], reg_m["asymmetric_penalty"],
    )

    log.info("Training complete. Run 'make evaluate' to print the comparison table.")


if __name__ == "__main__":
    main()
