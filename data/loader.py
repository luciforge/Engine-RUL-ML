"""CMAPSS data loader: parse raw .txt files into tidy DataFrames.

Column layout (26 columns per row, space-separated):
    1  unit_id          engine unit number
    2  cycle            time in cycles
    3–5  op_setting_1–3  operational settings
    6–26 sensor_1–21    sensor measurements
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

_VARIANTS = {"FD001", "FD002", "FD003", "FD004"}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _raw_dir() -> Path:
    # Allow env-var override for CI/Docker contexts
    env_override = os.environ.get("CMAPSS_RAW_DIR")
    if env_override:
        return Path(env_override).resolve()
    with open(_PROJECT_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return (_PROJECT_ROOT / cfg["data"]["raw_dir"]).resolve()


def _parse(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Drop trailing all-NaN column that appears in some variants with trailing spaces
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS[: len(df.columns)]
    return df


def load_fd(variant: str, split: str = "train") -> pd.DataFrame:
    """Load a CMAPSS variant split.

    Parameters
    ----------
    variant : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    split : str
        'train' (run-to-failure) or 'test' (truncated sequences).

    Returns
    -------
    pd.DataFrame with columns: unit_id, cycle, op_setting_1–3, sensor_1–21.
    """
    variant = variant.upper()
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    path = _raw_dir() / f"{split}_{variant}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = _parse(path)
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    try:
        from data.schemas.cmapss import cmapss_schema
        cmapss_schema.validate(df, lazy=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Schema validation warning for %s/%s: %s", variant, split, exc)

    logger.info(
        "Loaded %s/%s: %d rows, %d units", variant, split, len(df), df["unit_id"].nunique()
    )
    return df


def load_rul(variant: str) -> pd.Series:
    """Load ground-truth RUL vector for the test set.

    Returns a Series with 0-based index (engine index), values = true RUL at last cycle.
    """
    variant = variant.upper()
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")

    path = _raw_dir() / f"RUL_{variant}.txt"
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")

    rul = pd.read_csv(path, header=None, names=["rul"])["rul"]
    logger.info("Loaded RUL %s: %d engines", variant, len(rul))
    return rul
