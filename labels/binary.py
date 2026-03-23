"""Binary label: will the engine fail within X cycles from now?

Requires 'rul' column (call labels.rul.add_rul first).
X is read from config.yaml (labels.label_within_x, default 30).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_x() -> int:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return int(yaml.safe_load(f)["labels"]["label_within_x"])


def add_binary_label(df: pd.DataFrame, x: int | None = None) -> pd.DataFrame:
    """Add 'label_within_x' column: 1 if rul <= x, else 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'rul' column (produced by add_rul).
    x : int, optional
        Failure horizon in cycles. Defaults to config.yaml value (30).
    """
    if "rul" not in df.columns:
        raise ValueError("'rul' column required. Call labels.rul.add_rul() first.")
    if x is None:
        x = _default_x()
    out = df.copy()
    out["label_within_x"] = (out["rul"] <= x).astype(int)
    return out
