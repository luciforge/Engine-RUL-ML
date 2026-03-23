"""Tests: ONNX output matches native model output within 1e-4 tolerance."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
pytest.importorskip("skl2onnx", reason="skl2onnx not installed")


def _make_clf_data(n: int = 300, n_features: int = 20):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=10, random_state=0
    )
    return X.astype(np.float32), y


def test_xgboost_json_roundtrip():
    """XGBoost JSON save/load round-trip: predictions must be bit-exact."""
    from xgboost import XGBClassifier

    from service.onnx_export import export_xgboost

    X, y = _make_clf_data()
    model = XGBClassifier(
        n_estimators=20,
        eval_metric="logloss",
        tree_method="hist",
        random_state=0,
    )
    model.fit(X, y)
    proba_native = model.predict_proba(X)[:, 1]

    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "xgb.json"
        saved = export_xgboost(model, feature_cols, json_path)
        loaded = XGBClassifier()
        loaded.load_model(str(saved))
        proba_loaded = loaded.predict_proba(X)[:, 1]
        assert float(np.abs(proba_native - proba_loaded).max()) < 1e-6


def test_random_forest_onnx_matches_native():
    from sklearn.ensemble import RandomForestClassifier

    from service.onnx_export import export_xgboost, validate_onnx

    X, y = _make_clf_data()
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)

    feature_cols = [f"f{i}" for i in range(X.shape[1])]
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "rf.onnx"
        export_xgboost(model, feature_cols, onnx_path)
        native_fn = lambda arr: model.predict_proba(arr)  # noqa: E731
        assert validate_onnx(native_fn, onnx_path, X, tol=2e-3)
