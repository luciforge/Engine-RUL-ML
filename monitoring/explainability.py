"""SHAP-equivalent feature attribution for the XGBoost deployment classifier.

Uses XGBoost's native ``pred_contribs=True`` (C++ TreeShap) — no extra dependency.
Feature contributions are in log-odds space: positive → toward failure, negative → away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xgboost import XGBClassifier

_OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

#: Ordered feature names matching the 24-column deployment input layout.
FEATURE_NAMES: list[str] = _OP_COLS + _SENSOR_COLS


def shap_explain(
    model: "XGBClassifier",
    arr: np.ndarray,
    top_n: int = 10,
) -> tuple[dict[str, float], float]:
    """Return (attribution dict sorted by |SHAP|, base_value) for a single input row.

    attribution maps feature names to SHAP values in log-odds space.
    base_value + sum(all SHAP values) == raw model log-odds output.
    """
    import xgboost as xgb

    dmat = xgb.DMatrix(arr, feature_names=FEATURE_NAMES)
    # pred_contribs returns shape (n_samples, n_features + 1);
    # the last column is the bias term (base value).
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    row = contribs[0]
    shap_values = row[:-1]  # per-feature contributions
    base_value = float(row[-1])  # bias term

    order = np.argsort(np.abs(shap_values))[::-1][:top_n]
    attribution = {
        FEATURE_NAMES[i]: round(float(shap_values[i]), 6) for i in order
    }
    return attribution, base_value

