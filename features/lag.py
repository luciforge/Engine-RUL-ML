"""Lag features for sensor columns, computed per engine unit."""

from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    lag_k: list[int],
) -> pd.DataFrame:
    """Add lag-k features per unit_id (shift within each engine's time series)."""
    new_cols: dict[str, pd.Series] = {}
    for k in lag_k:
        for col in sensor_cols:
            new_cols[f"{col}_lag{k}"] = df.groupby("unit_id", group_keys=False)[col].transform(
                lambda s, _k=k: s.shift(_k)
            )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
