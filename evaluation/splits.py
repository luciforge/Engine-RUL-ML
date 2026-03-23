"""Two evaluation tracks for CMAPSS.

Track A — Intra-dataset unit-holdout (within each FD variant)
    - 80% of engine IDs → train, 20% → test (random split, fixed seed)
    - Run independently for FD001, FD002, FD003, FD004
    - Tests generalization to unseen engines under same operating conditions

Track B — Cross-domain operating-condition generalization
    - Train: FD001 + FD003  (1 operating condition, mixed fault modes)
    - Test:  FD002 + FD004  (6 operating conditions, mixed fault modes)
    - Tests robustness to operating-condition shift

Normalization rule (both tracks):
    Per-sensor z-score StandardScaler fit on training split only,
    then applied to test split — prevents leakage, stabilizes LSTM training.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from data.loader import load_fd
from features.pipeline import RollingLagTransformer, get_feature_cols
from labels.binary import add_binary_label
from labels.rul import add_rul

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


class SplitResult(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler
    feature_cols: list[str]


def _enrich(df: pd.DataFrame, x: int) -> pd.DataFrame:
    """Add RUL + binary label."""
    df = add_rul(df)
    df = add_binary_label(df, x)
    return df


def _normalize(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, apply to both. Returns (train, test, scaler)."""
    scaler = StandardScaler()
    train = train.copy()
    test = test.copy()
    train[feature_cols] = scaler.fit_transform(train[feature_cols].fillna(0.0))
    test[feature_cols] = scaler.transform(test[feature_cols].fillna(0.0))
    return train, test, scaler


def track_a_split(variant: str) -> SplitResult:
    """Track A: 80/20 engine-unit holdout within a single FD variant.

    Loads the run-to-failure training file, applies rolling+lag feature engineering,
    then randomly assigns 80% of unit_ids to train and 20% to test.
    Normalization is fit on train only.
    """
    cfg = _cfg()
    x = cfg["labels"]["label_within_x"]
    seed = cfg["training"]["random_seed"]
    ratio = cfg["training"]["train_engine_split"]

    df = load_fd(variant, split="train")
    df = _enrich(df, x)

    transformer = RollingLagTransformer()
    df = transformer.transform(df)

    unit_ids = df["unit_id"].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unit_ids)
    n_train = int(len(unit_ids) * ratio)
    train_ids = set(unit_ids[:n_train])
    test_ids = set(unit_ids[n_train:])

    train = df[df["unit_id"].isin(train_ids)].reset_index(drop=True)
    test = df[df["unit_id"].isin(test_ids)].reset_index(drop=True)

    feature_cols = get_feature_cols(train)
    train, test, scaler = _normalize(train, test, feature_cols)
    return SplitResult(train=train, test=test, scaler=scaler, feature_cols=feature_cols)


def track_b_split() -> SplitResult:
    """Track B: cross-domain operating-condition generalization.

    Train: FD001 + FD003  (single operating condition; mixed fault modes)
    Test:  FD002 + FD004  (six operating conditions; mixed fault modes)

    Uses run-to-failure training files for both sides so ground-truth RUL
    and binary labels are available for evaluation.
    """
    cfg = _cfg()
    x = cfg["labels"]["label_within_x"]

    transformer = RollingLagTransformer()

    train_parts: list[pd.DataFrame] = []
    for v in ("FD001", "FD003"):
        df = load_fd(v, split="train")
        df = _enrich(df, x)
        df = transformer.transform(df)
        df["variant"] = v
        train_parts.append(df)

    test_parts: list[pd.DataFrame] = []
    for v in ("FD002", "FD004"):
        df = load_fd(v, split="train")
        df = _enrich(df, x)
        df = transformer.transform(df)
        df["variant"] = v
        test_parts.append(df)

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    feature_cols = get_feature_cols(train)
    train, test, scaler = _normalize(train, test, feature_cols)
    return SplitResult(train=train, test=test, scaler=scaler, feature_cols=feature_cols)
