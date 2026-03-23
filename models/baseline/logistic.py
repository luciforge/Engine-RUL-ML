"""Logistic regression baseline on engineered features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as _LR

from features.pipeline import get_feature_cols


class LogisticBaseline:
    """Thin wrapper around sklearn LogisticRegression for consistency with other models."""

    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self._model = _LR(
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
            class_weight="balanced",
        )
        self._feature_cols: list[str] = []

    def fit(self, train: pd.DataFrame) -> "LogisticBaseline":
        self._feature_cols = get_feature_cols(train)
        X = train[self._feature_cols].fillna(0.0).values
        y = train["label_within_x"].values
        self._model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self._feature_cols].fillna(0.0).values
        return self._model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    @property
    def sklearn_model(self) -> _LR:
        return self._model

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)
