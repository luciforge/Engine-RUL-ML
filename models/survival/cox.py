"""CoxPH survival model with simulated right-censoring on CMAPSS training data.

Censoring strategy (right-censoring, statistically valid for survival analysis):
  - For each engine in the training set, with probability `censoring_ratio`
    the sequence is randomly truncated before the recorded failure cycle.
  - Truncated sequences: duration = cut_cycle, event = 0 (right-censored)
  - Complete sequences:  duration = max_cycle, event = 1 (observed failure)

This reflects realistic fleet data where some engines are still operational
at the time of data collection and have not yet reached failure (censored observations).

Primary evaluation metric: C-index (concordance index).
lifelines.CoxPHFitter.score() returns this directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cfg() -> dict:
    with open(_PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _build_survival_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    censoring_ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Convert per-cycle DataFrame to one row per engine for survival analysis.

    Features are taken from the last observed cycle of each engine.
    """
    rows = []
    for unit_id, grp in df.groupby("unit_id"):
        grp = grp.sort_values("cycle").reset_index(drop=True)
        max_cycle = int(grp["cycle"].max())

        if rng.random() < censoring_ratio and len(grp) > 3:
            # Random cut — at least 1 cycle observed, at most len-1
            cut_idx = int(rng.integers(1, len(grp)))
            grp = grp.iloc[:cut_idx]
            duration = int(grp["cycle"].max())
            event = 0
        else:
            duration = max_cycle
            event = 1

        row = grp.iloc[-1][feature_cols].infer_objects(copy=False).fillna(0.0).to_dict()
        row["duration"] = duration
        row["event"] = event
        rows.append(row)

    return pd.DataFrame(rows)


def train_cox(train: pd.DataFrame, feature_cols: list[str]) -> CoxPHFitter:
    """Fit CoxPHFitter on training data with simulated censoring.

    Low-variance features are dropped automatically to improve numerical stability.
    L2 penalizer=0.1 guards against multicollinearity from correlated sensors.
    """
    cfg = _cfg()
    censoring_ratio = cfg["survival"]["censoring_ratio"]
    seed = cfg["training"]["random_seed"]
    rng = np.random.default_rng(seed)

    surv_df = _build_survival_df(train, feature_cols, censoring_ratio, rng)

    # Drop near-zero-variance columns (rolling/lag features on short sequences)
    variances = surv_df[feature_cols].var()
    fit_cols = [c for c in feature_cols if variances.get(c, 0.0) >= 1e-8]

    # CoxPH is memory-intensive with hundreds of features — use a reduced set
    # (top-50 by variance) if the feature space is very large
    if len(fit_cols) > 50:
        fit_cols = list(variances[fit_cols].nlargest(50).index)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        surv_df[fit_cols + ["duration", "event"]],
        duration_col="duration",
        event_col="event",
    )
    return cph


def evaluate_c_index(
    cph: CoxPHFitter,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    """Compute C-index on the test set using the same engine-level representation."""
    cfg = _cfg()
    seed = cfg["training"]["random_seed"] + 1  # different seed for test censoring
    rng = np.random.default_rng(seed)

    surv_df = _build_survival_df(
        test, feature_cols, cfg["survival"]["censoring_ratio"], rng
    )

    # Use only features present in the fitted model
    fit_cols = [c for c in cph.params_.index if c in surv_df.columns]
    surv_valid = surv_df[fit_cols + ["duration", "event"]].dropna()

    return float(cph.score(surv_valid, scoring_method="concordance_index"))
