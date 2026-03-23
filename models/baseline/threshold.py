"""Data-driven threshold rule baseline.

Sensor selection via Spearman rank correlation with RUL on the training set:
the sensor with the highest |Spearman ρ| against RUL is the most monotonically
degrading signal and becomes the classifier trigger.

No sensor numbers are hardcoded — selection is fully data-driven.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def select_degrading_sensor(train: pd.DataFrame, sensor_cols: list[str]) -> str:
    """Return the sensor column most monotonically correlated with RUL.

    Uses absolute Spearman ρ to capture both rising (positive) and falling
    (negative) degradation trends equally.
    """
    if "rul" not in train.columns:
        raise ValueError("'rul' column required in train DataFrame")

    correlations: dict[str, float] = {}
    for col in sensor_cols:
        valid = train[[col, "rul"]].dropna()
        if len(valid) < 10:
            correlations[col] = 0.0
            continue
        rho, _ = spearmanr(valid[col].values, valid["rul"].values)
        correlations[col] = abs(float(rho))

    return max(correlations, key=correlations.__getitem__)


class ThresholdClassifier:
    """Binary classifier: fires when chosen sensor crosses a learned threshold.

    Threshold is set as (mean ± std) of the sensor on training positive rows
    (label_within_x == 1), with direction determined by class separation.
    """

    def __init__(self):
        self.sensor_col_: str | None = None
        self.threshold_: float | None = None
        self._direction: int = 1  # +1 → high value = failure; -1 → low value = failure

    def fit(self, train: pd.DataFrame, sensor_cols: list[str]) -> "ThresholdClassifier":
        self.sensor_col_ = select_degrading_sensor(train, sensor_cols)

        positives = train[train["label_within_x"] == 1][self.sensor_col_].dropna()
        negatives = train[train["label_within_x"] == 0][self.sensor_col_].dropna()

        if positives.mean() >= negatives.mean():
            self._direction = 1
            self.threshold_ = float(positives.mean() - positives.std())
        else:
            self._direction = -1
            self.threshold_ = float(positives.mean() + positives.std())
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        vals = df[self.sensor_col_].fillna(df[self.sensor_col_].median()).values
        vmin, vmax = vals.min(), vals.max()
        denom = (vmax - vmin) + 1e-9
        if self._direction == 1:
            scores = (vals - vmin) / denom
        else:
            scores = (vmax - vals) / denom
        return scores.astype(float)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)
