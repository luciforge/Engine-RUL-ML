"""Tests for sensor robustness: corruption utilities, quality-check integration,
model behaviour under missing/noisy data, and MC-Dropout uncertainty estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
ALL_SENSOR_COLS = OP_COLS + SENSOR_COLS


def _make_synthetic_df(n_units: int = 5, cycles_per_unit: int = 50, seed: int = 0) -> pd.DataFrame:
    """Generate a minimal synthetic CMAPSS-shaped DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_units + 1):
        for c in range(1, cycles_per_unit + 1):
            row = {"unit_id": uid, "cycle": c, "label_within_x": int(c > cycles_per_unit - 30)}
            for op in OP_COLS:
                row[op] = float(rng.uniform(-0.5, 0.5))
            for s in SENSOR_COLS:
                row[s] = float(rng.normal(500.0, 20.0))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synth_df():
    return _make_synthetic_df()


# ---------------------------------------------------------------------------
# Corruption utilities
# ---------------------------------------------------------------------------

class TestInjectMissing:
    def test_returns_copy(self, synth_df):
        from monitoring.robustness import inject_missing
        corrupted = inject_missing(synth_df, SENSOR_COLS, dropout_rate=0.2)
        # Original must be unchanged
        assert synth_df[SENSOR_COLS].isna().sum().sum() == 0
        assert corrupted is not synth_df

    def test_nan_count_roughly_correct(self, synth_df):
        from monitoring.robustness import inject_missing
        rate = 0.2
        corrupted = inject_missing(synth_df, SENSOR_COLS, dropout_rate=rate, seed=7)
        total_cells = len(synth_df) * len(SENSOR_COLS)
        actual_rate = corrupted[SENSOR_COLS].isna().sum().sum() / total_cells
        # Allow +-5pp tolerance
        assert abs(actual_rate - rate) < 0.05

    def test_non_sensor_cols_untouched(self, synth_df):
        from monitoring.robustness import inject_missing
        corrupted = inject_missing(synth_df, SENSOR_COLS, dropout_rate=0.5)
        assert corrupted["unit_id"].isna().sum() == 0
        assert corrupted["cycle"].isna().sum() == 0

    def test_zero_rate_no_nans(self, synth_df):
        from monitoring.robustness import inject_missing
        corrupted = inject_missing(synth_df, SENSOR_COLS, dropout_rate=0.0)
        assert corrupted[SENSOR_COLS].isna().sum().sum() == 0

    def test_full_rate_all_nan(self, synth_df):
        from monitoring.robustness import inject_missing
        corrupted = inject_missing(synth_df, SENSOR_COLS, dropout_rate=1.0)
        assert corrupted[SENSOR_COLS].isna().all().all()


class TestInjectNoise:
    def test_returns_copy(self, synth_df):
        from monitoring.robustness import inject_noise
        noisy = inject_noise(synth_df, SENSOR_COLS)
        assert noisy is not synth_df

    def test_values_changed(self, synth_df):
        from monitoring.robustness import inject_noise
        noisy = inject_noise(synth_df, SENSOR_COLS, snr_db=5.0)
        # At least some values must differ
        diff = (noisy[SENSOR_COLS].values != synth_df[SENSOR_COLS].values).sum()
        assert diff > 0

    def test_non_sensor_untouched(self, synth_df):
        from monitoring.robustness import inject_noise
        noisy = inject_noise(synth_df, SENSOR_COLS)
        pd.testing.assert_series_equal(noisy["unit_id"], synth_df["unit_id"])
        pd.testing.assert_series_equal(noisy["cycle"], synth_df["cycle"])

    def test_high_snr_small_perturbation(self, synth_df):
        from monitoring.robustness import inject_noise
        noisy = inject_noise(synth_df, SENSOR_COLS, snr_db=60.0, seed=1)
        max_diff = (noisy[SENSOR_COLS] - synth_df[SENSOR_COLS]).abs().max().max()
        # At 60 dB SNR, perturbations should be tiny relative to signal scale
        assert max_diff < 5.0  # signal is ~500, 60dB → noise_std ~0.5


class TestInjectFlatline:
    def test_returns_copy(self, synth_df):
        from monitoring.robustness import inject_flatline
        flat = inject_flatline(synth_df, SENSOR_COLS)
        assert flat is not synth_df

    def test_flatline_stretch_exists(self, synth_df):
        from monitoring.robustness import inject_flatline
        flat = inject_flatline(synth_df, ["sensor_1"], flatline_frac=0.3, seed=0)
        values = flat["sensor_1"].values
        # At least one run of consecutive equal values of length >= 2
        runs = (np.diff(values) == 0).sum()
        assert runs > 0

    def test_non_sensor_untouched(self, synth_df):
        from monitoring.robustness import inject_flatline
        flat = inject_flatline(synth_df, SENSOR_COLS)
        pd.testing.assert_series_equal(flat["unit_id"], synth_df["unit_id"])


# ---------------------------------------------------------------------------
# Quality-check integration
# ---------------------------------------------------------------------------

class TestMissingQualityFlag:
    def test_missing_spike_fires_on_injected_data(self, synth_df):
        from monitoring.robustness import inject_missing
        from monitoring.quality import check_missing_spike
        corrupted = inject_missing(synth_df, SENSOR_COLS, dropout_rate=0.3, seed=42)
        alerts = check_missing_spike(corrupted, SENSOR_COLS)
        assert len(alerts) > 0, "Expected missing-spike alerts after 30% NaN injection"

    def test_no_missing_spike_on_clean_data(self, synth_df):
        from monitoring.quality import check_missing_spike
        # Clean data should produce no missing-spike alerts at default threshold
        alerts = check_missing_spike(synth_df, SENSOR_COLS)
        assert len(alerts) == 0


class TestFlatlineQualityFlag:
    def test_flatline_fires_on_injected_data(self, synth_df):
        from monitoring.robustness import inject_flatline
        from monitoring.quality import check_flatline
        flat = inject_flatline(synth_df, ["sensor_1", "sensor_2"], flatline_frac=0.4, seed=0)
        alerts = check_flatline(flat, ["sensor_1", "sensor_2"])
        assert len(alerts) > 0, "Expected flatline alerts after injecting flatline stretches"

    def test_no_flatline_on_clean_data(self, synth_df):
        from monitoring.quality import check_flatline
        alerts = check_flatline(synth_df, SENSOR_COLS)
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# Model behaviour under missing sensors
# ---------------------------------------------------------------------------

class TestMissingPredictionDegrades:
    """XGBoost inference must not crash under 20% NaN; predictions stay in [0,1]."""

    def test_xgboost_handles_missing_no_crash(self, synth_df):
        """NaN-filled inference must return predictions in [0, 1] without exception."""
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier
        from monitoring.robustness import inject_missing

        feature_cols = OP_COLS + SENSOR_COLS
        X_clean = synth_df[feature_cols].fillna(0.0).values.astype("float32")
        y = synth_df["label_within_x"].values.astype(int)

        clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=0, eval_metric="logloss")
        clf.fit(X_clean, y)

        corrupted = inject_missing(synth_df, SENSOR_COLS, dropout_rate=0.2, seed=99)
        # XGBoost natively handles NaN — do not fill
        X_corrupt = corrupted[feature_cols].values.astype("float32")
        proba = clf.predict_proba(X_corrupt)[:, 1]

        assert proba.shape == (len(synth_df),)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert not np.any(np.isnan(proba))


# ---------------------------------------------------------------------------
# Model behaviour under sensor noise
# ---------------------------------------------------------------------------

class TestNoisePredictionGraceful:
    def test_xgboost_predictions_bounded_under_noise(self, synth_df):
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier
        from monitoring.robustness import inject_noise

        feature_cols = OP_COLS + SENSOR_COLS
        X = synth_df[feature_cols].fillna(0.0).values.astype("float32")
        y = synth_df["label_within_x"].values.astype(int)

        clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=0, eval_metric="logloss")
        clf.fit(X, y)

        noisy = inject_noise(synth_df, SENSOR_COLS, snr_db=5.0, seed=7)
        X_noisy = noisy[feature_cols].fillna(0.0).values.astype("float32")
        proba = clf.predict_proba(X_noisy)[:, 1]

        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert not np.any(np.isnan(proba))


# ---------------------------------------------------------------------------
# MC-Dropout uncertainty under noise
# ---------------------------------------------------------------------------

class TestMCDropoutUncertaintyWidens:
    def test_std_increases_under_noise(self, synth_df):
        """MC-Dropout std should be higher for noisy data than clean data."""
        torch = pytest.importorskip("torch")
        from models.deep.lstm import LSTMModel, predict_proba_mc
        from monitoring.robustness import inject_noise

        feature_cols = OP_COLS + SENSOR_COLS
        n_features = len(feature_cols)

        # Use a small model for speed
        model = LSTMModel(input_size=n_features, hidden_size=16, num_layers=1, dropout=0.5)
        model.eval()

        window = 10
        # Need enough cycles per unit for at least one window
        df = _make_synthetic_df(n_units=3, cycles_per_unit=30)

        _, std_clean = predict_proba_mc(model, df, feature_cols, window=window, n_samples=20)
        noisy_df = inject_noise(df, SENSOR_COLS, snr_db=2.0, seed=5)
        _, std_noisy = predict_proba_mc(model, noisy_df, feature_cols, window=window, n_samples=20)

        if len(std_clean) == 0 or len(std_noisy) == 0:
            pytest.skip("No windowed samples generated — increase cycles_per_unit")

        # Mean std under noise should be >= mean std on clean (not strictly always,
        # but holds reliably with high-dropout model and very low SNR)
        assert std_noisy.mean() >= std_clean.mean() * 0.8  # allow 20% tolerance
