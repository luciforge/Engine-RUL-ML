"""Tests: pandera schema accepts valid input and rejects malformed data."""

from __future__ import annotations

import pandas as pd
import pytest

from data.schemas.cmapss import validate


def _base_row() -> dict:
    return {
        "unit_id": 1,
        "cycle": 1,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
        **{f"sensor_{i}": 100.0 for i in range(1, 22)},
    }


def test_valid_single_row():
    df = pd.DataFrame([_base_row()])
    result = validate(df)
    assert result is not None
    assert len(result) == 1


def test_valid_multi_unit():
    rows = []
    for unit in [1, 2, 3]:
        for cycle in [1, 2, 3]:
            r = _base_row()
            r["unit_id"] = unit
            r["cycle"] = cycle
            rows.append(r)
    result = validate(pd.DataFrame(rows))
    assert len(result) == 9


def test_rejects_negative_unit_id():
    row = _base_row()
    row["unit_id"] = -1
    with pytest.raises(Exception):
        validate(pd.DataFrame([row]))


def test_rejects_zero_cycle():
    row = _base_row()
    row["cycle"] = 0
    with pytest.raises(Exception):
        validate(pd.DataFrame([row]))


def test_rejects_op_setting_3_out_of_range():
    row = _base_row()
    row["op_setting_3"] = 200.0  # valid range is 0–100
    with pytest.raises(Exception):
        validate(pd.DataFrame([row]))


def test_rejects_duplicate_unit_cycle():
    row = _base_row()
    df = pd.DataFrame([row, row])  # exact duplicate → same (unit_id, cycle)
    with pytest.raises(Exception):
        validate(df)


def test_rejects_all_nan_row():
    row = _base_row()
    nan_row = {k: float("nan") for k in row}
    with pytest.raises(Exception):
        validate(pd.DataFrame([nan_row]))
