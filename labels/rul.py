"""RUL label: remaining useful life = max_cycle − current_cycle per engine unit.

At the final recorded cycle of each engine, RUL = 0.
Only applicable to training data (run-to-failure sequences).
"""

from __future__ import annotations

import pandas as pd


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'rul' column to a run-to-failure DataFrame.

    RUL = max_cycle_for_unit − current_cycle.
    At the last recorded cycle, RUL = 0.
    """
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    out = df.copy()
    out["rul"] = (max_cycles - out["cycle"]).astype(int)
    return out
