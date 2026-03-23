"""Tests: RUL and binary label correctness."""

from __future__ import annotations

import pandas as pd
import pytest

from labels.binary import add_binary_label
from labels.rul import add_rul


def _make_df(units: list[int] = (1, 2), max_cycle: int = 10) -> pd.DataFrame:
    rows = []
    for unit in units:
        for cycle in range(1, max_cycle + 1):
            row = {
                "unit_id": unit,
                "cycle": cycle,
                **{f"sensor_{i}": 1.0 for i in range(1, 22)},
                **{f"op_setting_{i}": 0.0 for i in range(1, 4)},
            }
            rows.append(row)
    return pd.DataFrame(rows)



def test_rul_at_last_cycle_is_zero():
    df = add_rul(_make_df())
    last = df.groupby("unit_id").apply(lambda g: g.loc[g["cycle"].idxmax(), "rul"], include_groups=False)
    assert (last == 0).all(), "RUL must be 0 at the last cycle of every engine"


def test_rul_at_first_cycle():
    df = add_rul(_make_df(max_cycle=10))
    first = df[df["cycle"] == 1]["rul"]
    # max_cycle=10, current=1 → RUL = 9
    assert (first == 9).all()


def test_rul_monotone_decreasing_per_unit():
    df = add_rul(_make_df())
    for _, grp in df.groupby("unit_id"):
        rul_vals = grp.sort_values("cycle")["rul"].values
        assert (rul_vals[:-1] >= rul_vals[1:]).all(), (
            "RUL must be non-increasing along the cycle axis"
        )


def test_rul_independent_per_unit():
    """Two engines with different max cycles must have independent RUL counts."""
    rows = (
        [{"unit_id": 1, "cycle": c, **{f"sensor_{i}": 0.0 for i in range(1, 22)},
          **{f"op_setting_{i}": 0.0 for i in range(1, 4)}} for c in range(1, 6)]
        + [{"unit_id": 2, "cycle": c, **{f"sensor_{i}": 0.0 for i in range(1, 22)},
            **{f"op_setting_{i}": 0.0 for i in range(1, 4)}} for c in range(1, 11)]
    )
    df = add_rul(pd.DataFrame(rows))
    assert df[df["unit_id"] == 1]["rul"].max() == 4   # max_cycle=5
    assert df[df["unit_id"] == 2]["rul"].max() == 9   # max_cycle=10



def test_binary_label_requires_rul_column():
    df = _make_df()
    with pytest.raises(ValueError, match="rul"):
        add_binary_label(df, x=5)


def test_binary_label_values():
    df = add_rul(_make_df(max_cycle=10))
    df = add_binary_label(df, x=3)
    # rul=0,1,2,3 → label=1 (cycles 10, 9, 8, 7)
    # rul=4..9    → label=0 (cycles 6, 5, 4, 3, 2, 1)
    labeled_cycles = df[df["label_within_x"] == 1]["cycle"].unique()
    unlabeled_cycles = df[df["label_within_x"] == 0]["cycle"].unique()
    assert 10 in labeled_cycles
    assert 7 in labeled_cycles
    assert 6 in unlabeled_cycles
    assert 1 in unlabeled_cycles


def test_binary_label_no_positives_when_x_negative():
    df = add_rul(_make_df())
    df = add_binary_label(df, x=-1)
    assert df["label_within_x"].sum() == 0
