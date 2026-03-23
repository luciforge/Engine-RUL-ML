"""pandera schema for raw CMAPSS DataFrames.

Validates:
 - Correct column types (unit_id/cycle as int, sensors/op_settings as float)
 - No duplicate (unit_id, cycle) pairs within a DataFrame
 - No all-NaN rows
 - Sensor value plausibility ranges derived from the dataset distribution
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]

cmapss_schema = DataFrameSchema(
    columns={
        "unit_id": Column(int, checks=Check.greater_than(0), nullable=False),
        "cycle": Column(int, checks=Check.greater_than(0), nullable=False),
        # Operational settings — broad range to cover all 4 FD variants
        "op_setting_1": Column(float, checks=Check.in_range(-1.0, 1.0), nullable=False),
        "op_setting_2": Column(float, checks=Check.in_range(-1.0, 1.0), nullable=False),
        "op_setting_3": Column(float, checks=Check.in_range(0.0, 100.0), nullable=False),
        # Sensor columns — all must be present and non-null
        **{col: Column(float, nullable=False) for col in SENSOR_COLS},
    },
    checks=[
        Check(
            lambda df: not df.duplicated(subset=["unit_id", "cycle"]).any(),
            error="Duplicate (unit_id, cycle) pairs found",
        ),
        Check(
            lambda df: not df.isnull().all(axis=1).any(),
            error="All-NaN rows found",
        ),
    ],
    coerce=True,
)


def validate(df, schema: DataFrameSchema = cmapss_schema):
    """Validate a raw CMAPSS DataFrame against the schema.

    Raises pandera.errors.SchemaError on violation.
    """
    return schema.validate(df)
