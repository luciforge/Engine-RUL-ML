"""Rolling window features: mean, std, and linear slope per sensor.

All computations are done per unit_id to prevent cross-engine leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(series: pd.Series) -> float:
    """Least-squares slope of a rolling window (used via Series.apply)."""
    y = series.values
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_sizes: list[int],
) -> pd.DataFrame:
    """Add rolling mean / std / slope features for each sensor and window size.

    min_periods = window // 2 to limit NaNs at the start of short sequences.
    Features are computed independently per unit_id.
    """
    new_cols: dict[str, pd.Series] = {}
    for w in window_sizes:
        mp = max(1, w // 2)
        for col in sensor_cols:
            grp = df.groupby("unit_id", group_keys=False)[col]
            new_cols[f"{col}_mean_w{w}"] = grp.transform(
                lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).mean()
            )
            new_cols[f"{col}_std_w{w}"] = grp.transform(
                lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).std()
            )
            new_cols[f"{col}_slope_w{w}"] = grp.transform(
                lambda s, _w=w, _mp=mp: s.rolling(_w, min_periods=_mp).apply(
                    _slope, raw=False
                )
            )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
