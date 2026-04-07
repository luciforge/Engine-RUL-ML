"""Sensor robustness corruption utilities for stress-testing PdM models.

Three synthetic failure modes that reflect real industrial sensor degradation:

  inject_missing   — random NaN dropout (sensor disconnection / data loss)
  inject_noise     — additive Gaussian noise at a specified SNR (sensor degradation)
  inject_flatline  — contiguous constant-value stretches (stuck sensor / frozen reading)

All functions operate on a copy of the input DataFrame and do not modify it in place.
The ``sensor_cols`` argument controls which columns are corrupted; other columns are untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_missing(
    df: pd.DataFrame,
    sensor_cols: list[str],
    dropout_rate: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly NaN-out sensor readings to simulate data loss or disconnection.

    Each (row, sensor) pair is independently set to NaN with probability
    ``dropout_rate``.

    Parameters
    ----------
    df           : input DataFrame (not modified in place)
    sensor_cols  : columns to corrupt
    dropout_rate : fraction of values to drop, in [0, 1]
    seed         : random seed for reproducibility

    Returns
    -------
    Copy of ``df`` with NaNs injected into ``sensor_cols``.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in sensor_cols:
        if col not in out.columns:
            continue
        mask = rng.random(size=len(out)) < dropout_rate
        out.loc[mask, col] = np.nan
    return out


def inject_noise(
    df: pd.DataFrame,
    sensor_cols: list[str],
    snr_db: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Add Gaussian noise at a specified signal-to-noise ratio (dB).

    SNR_dB = 10 * log10(signal_power / noise_power)
    Noise standard deviation = signal_std / 10^(SNR_dB / 20)

    Parameters
    ----------
    df          : input DataFrame (not modified in place)
    sensor_cols : columns to corrupt
    snr_db      : target signal-to-noise ratio in decibels (lower = more noise)
    seed        : random seed for reproducibility

    Returns
    -------
    Copy of ``df`` with additive Gaussian noise applied to ``sensor_cols``.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in sensor_cols:
        if col not in out.columns:
            continue
        signal = out[col].dropna().values.astype(float)
        if len(signal) == 0:
            continue
        signal_std = float(np.std(signal))
        if signal_std < 1e-10:
            # Constant column — add absolute noise floor of 0.01
            noise_std = 0.01
        else:
            noise_std = signal_std / (10.0 ** (snr_db / 20.0))
        noise = rng.normal(0.0, noise_std, size=len(out))
        out[col] = out[col] + noise
    return out


def inject_flatline(
    df: pd.DataFrame,
    sensor_cols: list[str],
    flatline_frac: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace random contiguous stretches with a constant (stuck-sensor simulation).

    For each sensor column, picks a random contiguous window covering
    approximately ``flatline_frac`` of the rows and replaces those values
    with the value at the start of the window (simulating a frozen reading).

    Parameters
    ----------
    df            : input DataFrame (not modified in place)
    sensor_cols   : columns to corrupt
    flatline_frac : approximate fraction of rows to flatline, in (0, 1)
    seed          : random seed for reproducibility

    Returns
    -------
    Copy of ``df`` with flatline segments injected into ``sensor_cols``.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    window_len = max(1, int(n * flatline_frac))

    for col in sensor_cols:
        if col not in out.columns:
            continue
        start = int(rng.integers(0, max(1, n - window_len + 1)))
        end = min(start + window_len, n)
        frozen_value = out[col].iloc[start]
        out.iloc[start:end, out.columns.get_loc(col)] = frozen_value

    return out
