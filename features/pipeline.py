"""Feature engineering pipeline.

RollingLagTransformer: adds rolling-window (mean/std/slope) and lag-k features.
Stateless — safe to apply identically to train and test splits.

Normalisation is handled downstream in evaluation/splits._normalize() using
StandardScaler fit on the training partition only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

from features.lag import add_lag_features
from features.rolling import add_rolling_features

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
_META_COLS = {"unit_id", "cycle", "rul", "label_within_x", "variant"}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


class RollingLagTransformer(BaseEstimator, TransformerMixin):
    """Adds rolling (mean/std/slope) and lag-k features.

    Stateless — can be applied identically to train and test splits.
    Expects a full DataFrame with 'unit_id' column for per-engine grouping.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        lag_k: list[int] | None = None,
    ):
        cfg = _load_cfg()
        self.window_sizes = window_sizes or cfg["features"]["window_sizes"]
        self.lag_k = lag_k or cfg["features"]["lag_k"]

    def fit(self, df: pd.DataFrame, y=None) -> "RollingLagTransformer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_rolling_features(df, SENSOR_COLS, self.window_sizes)
        df = add_lag_features(df, SENSOR_COLS, self.lag_k)
        return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature columns — all columns except metadata/labels."""
    return [c for c in df.columns if c not in _META_COLS]
