# Predictive Maintenance System — NASA CMAPSS FD001–FD004

End-to-end ML engineering system for turbofan engine failure prediction.  
Two evaluation tracks, five model types, MLflow registry, XGBoost native serving, Evidently monitoring, Docker Compose.

---

## Dataset

| Variant | Operating conditions | Fault modes | Train engines | Test engines |
|---------|---------------------|-------------|---------------|--------------|
| FD001   | 1 (Sea Level)       | 1 (HPC)     | 100           | 100          |
| FD002   | 6                   | 1 (HPC)     | 260           | 259          |
| FD003   | 1 (Sea Level)       | 2 (HPC+Fan) | 100           | 100          |
| FD004   | 6                   | 2 (HPC+Fan) | 248           | 249          |

Each row: `unit_id`, `cycle`, `op_setting_1–3`, `sensor_1–21` (26 columns, space-separated).

---

## Evaluation Tracks

### Track A — Intra-dataset unit-holdout
80% of engine IDs → train, 20% → test, independently per FD variant.  
Tests generalisation to unseen engines under the same operating conditions.

### Track B — Cross-domain operating-condition shift
```
Train: FD001 + FD003   (1 operating condition; mixed fault modes)
Test:  FD002 + FD004   (6 operating conditions; mixed fault modes)
```
Tests robustness to operating-condition shift — the defining challenge across CMAPSS variants.

**Normalisation:** per-sensor z-score StandardScaler fit on training split only, applied to test. Prevents leakage, stabilises LSTM training.

---

## Model Results

> All results below are **measured values** from make train && make evaluate && make benchmark.

### Classification (label_within_30) — PR-AUC

| Model              | Track A (FD001) | Track A (FD002) | Track A (FD003) | Track A (FD004) | Track B¹ |
|--------------------|-----------------|-----------------|-----------------|-----------------|----------|
| Threshold rule     | 0.151 | 0.442 | 0.127 | 0.531 | 0.137 |
| Logistic Regression| 0.976 | 0.958 | 0.986 | 0.924 | 0.140 |
| Random Forest      | 0.980 | 0.916 | 0.975 | 0.895 | 0.171 |
| **XGBoost + HPO**  | **0.980** | **0.928** | **0.985** | **0.923** | **0.221** |
| PyTorch LSTM       | 0.980 | 0.770 | **0.988** | 0.814 | — |

¹ Track B trains on FD001+FD003 (1 operating condition) and tests on FD002+FD004 (6 operating conditions). All classifiers decline sharply under operating-condition shift — XGBoost retains the highest Track B PR-AUC. LSTM is trained per-variant only and has no Track B result.

### Recall @ Precision=0.8 (FD001 test set)

| Model              | Track A | Track B¹ |
|--------------------|---------|----------|
| Threshold rule     | 0.000   | 0.000    |
| Logistic Regression| 0.971   | 0.000    |
| Random Forest      | 0.990   | 0.001    |
| XGBoost + HPO      | 0.979   | 0.029    |
| PyTorch LSTM       | 0.992   | —        |

### RUL Regression — XGBoost Deployment Regressor (FD001 test set, raw 24 features)

| Model | MAE | RMSE | Asymmetric Penalty |
|-------|-----|------|--------------------|
| XGBoost deployment regressor | 68.7 | 81.0 | 101.6 |

### Survival Analysis — C-index (CoxPH only)

| Variant | Track A | Track B |
|---------|---------|---------|
| FD001   | 0.724   | 0.517   |
| FD002   | 0.573   | 0.493   |
| FD003   | 0.717   | 0.487   |
| FD004   | 0.409   | 0.490   |

### Calibration — Brier Score / ECE (binary classifiers, FD001 test set)

| Model              | Brier (A) | ECE (A) | Brier (B) | ECE (B) |
|--------------------|-----------|---------|-----------|---------|
| Logistic Regression| 0.025     | 0.022   | 0.422     | 0.422   |
| Random Forest      | 0.018     | 0.017   | 0.136     | 0.139   |
| XGBoost + HPO      | 0.021     | 0.019   | 0.364     | 0.386   |
| PyTorch LSTM       | 0.024     | 0.020   | —         | —       |

All classifiers are well-calibrated on Track A (Brier < 0.03, ECE < 0.025). Track B calibration collapses under operating-condition shift — confirming PR-AUC degradation is not a threshold artefact.

### Latency Benchmark — Batch size 1, 1000 runs

| Model                              | P50     | P95     | P99     |
|------------------------------------|---------|---------|---------|
| Logistic regression (276 features) | 0.11 ms | 0.12 ms | 0.15 ms |
| XGBoost deployment (24 features)   | 0.71 ms | 1.04 ms | 1.87 ms |


Logistic dot-product is faster than XGBoost tree-traversal at batch size 1. At larger batch sizes XGBoost amortises the tree overhead and closes the gap.
---

## Repo Structure

```
predictive-maintenance/
├── data/
│   ├── loader.py            # load_fd(), load_rul()
│   └── schemas/cmapss.py    # pandera validation
├── features/
│   ├── rolling.py           # rolling mean/std/slope (w=10/20/30)
│   ├── lag.py               # lag-k features (k=1,3,5)
│   └── pipeline.py          # RollingLagTransformer
├── labels/
│   ├── rul.py               # max_cycle − current_cycle
│   └── binary.py            # label_within_X (default X=30)
├── models/
│   ├── baseline/            # ThresholdClassifier + LogisticBaseline
│   ├── classical/           # RandomForest + XGBoost + Optuna HPO
│   ├── deep/lstm.py         # PyTorch LSTM + SlidingWindowDataset
│   └── survival/cox.py      # CoxPHFitter with simulated censoring
├── evaluation/
│   ├── splits.py            # track_a_split() + track_b_split()
│   └── metrics.py           # PR-AUC, F1, MAE, C-index, Brier, ECE
├── policy/
│   └── notification.py      # ServiceAlert + evaluate_alert (urgency + service date) + threshold sweep
├── mlops/
│   └── tracking.py          # MLflow run logging
├── service/
│   ├── api.py               # FastAPI: /health, /predict, /explain, /schedule, /batch_score
│   ├── schemas.py           # Pydantic v2 request/response
│   ├── onnx_export.py       # XGBoost/LSTM → ONNX + validation
│   └── benchmark.py         # p50/p95/p99 native vs ONNX
├── monitoring/
│   ├── drift.py             # evidently DataDriftPreset
│   ├── quality.py           # flatline + missing spike detection
│   └── explainability.py   # XGBoost native TreeShap (pred_contribs) — no extra dependency
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_policy_simulation.ipynb
├── scripts/
│   ├── train.py             # python -m scripts.train
│   ├── evaluate.py          # python -m scripts.evaluate
│   ├── benchmark.py         # python -m scripts.benchmark
│   └── drift.py             # python -m scripts.drift
├── tests/
│   ├── test_schema.py       # pandera schema validation
│   ├── test_labels.py       # RUL correctness
│   ├── test_metrics.py      # metric function correctness
│   ├── test_notification.py # ServiceAlert urgency + scheduling logic
│   ├── test_onnx.py         # ONNX fidelity (tol=1e-4)
│   └── test_api.py          # FastAPI endpoint contracts
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.mlflow
│   └── docker-compose.yml   # api + mlflow + postgres
├── config.yaml
├── Makefile
└── pyproject.toml
```

---

## Design Decisions

**Why two evaluation tracks?**  
Track A answers "does the model generalise to unseen engines?" Track B answers "does it survive domain shift?" — the harder, more interview-relevant question. Reporting both prevents cherry-picking on the easier track.

**Why simulate censoring for CoxPH?**  
Real fleet data always has censored observations (engines still running). Training CoxPH on only fully observed failures biases the hazard estimates. The 30% random censoring approximates realistic fleet snapshot conditions.

**Why XGBoost JSON serving (not ONNX)?**  
XGBoost 3.x removed ONNX export support. The native `.json` format is the stable round-trip serialisation and loads directly into `XGBClassifier.load_model()` with no runtime conversion overhead. `onnx_export.py` still exports sklearn models (LogisticRegression, RandomForest) to ONNX for fidelity testing, but the FastAPI service loads XGBoost JSON exclusively.

**Why Postgres for MLflow?**  
SQLite does not support the MLflow Model Registry API (requires a database backend). Postgres in Docker Compose provides a production-equivalent registry without infrastructure overhead.

**Why Spearman over Pearson for sensor selection?**  
Sensor degradation is monotone but non-linear. Spearman captures any monotonic trend; Pearson would miss non-linear degradation curves.

---

## How to Run

### Local development

```bash
# 1. Install
make install

# 2. Set local MLflow URI (SQLite — no Postgres needed)
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db   # Linux/Mac
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"   # PowerShell

# 3. Train all models
make train

# 4. Print results table
make evaluate

# 5. Start API
make serve          # http://localhost:8000/docs

# 6a. Explain a prediction (SHAP attributions)
# POST http://localhost:8000/explain  (same payload as /predict)
# Returns top-10 feature contributions with risk_score

# 6b. Get maintenance schedule recommendation
# POST http://localhost:8000/schedule  (same payload, optional ?cycles_per_day=3.0)
# Returns urgency level + recommended service date

# 6. Run benchmarks
make benchmark

# 7. Drift report
make drift-report   # → reports/drift/drift_report.html

# 8. Tests
make test
```

### Docker Compose (full stack)

```bash
# Train models first
make train

# Start api + mlflow + postgres
cd docker
docker compose up --build

# Verify
curl http://localhost:8000/health
curl http://localhost:5000/health
```

### Sample /predict payload

The same payload works for `POST /explain`, `POST /schedule` — all three share the same `SensorInput` schema.


```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": 1, "cycle": 150,
    "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0,
    "sensor_1": 518.67, "sensor_2": 642.0, "sensor_3": 1585.0,
    "sensor_4": 1400.0, "sensor_5": 14.62, "sensor_6": 21.61,
    "sensor_7": 554.0, "sensor_8": 2388.0, "sensor_9": 9046.0,
    "sensor_10": 1.3, "sensor_11": 47.5, "sensor_12": 521.7,
    "sensor_13": 2388.0, "sensor_14": 8138.0, "sensor_15": 8.42,
    "sensor_16": 0.03, "sensor_17": 392, "sensor_18": 2388,
    "sensor_19": 100.0, "sensor_20": 39.0, "sensor_21": 23.4
  }'
```

Expected response:
```json
{
  "unit_id": 1,
  "risk_score": 0.31,
  "replace_within_30": false,
  "estimated_rul": 86.25
}
```

`POST /schedule` response (same payload, `?cycles_per_day=3.0`):
```json
{
  "unit_id": 1,
  "risk_score": 0.31,
  "estimated_rul_cycles": 86.25,
  "urgency": "scheduled",
  "recommended_service_date": "2026-04-22",
  "days_until_service": 29,
  "message": "Unit 1: EARLY WARNING — plan service within schedule. Estimated RUL 86 cycles (29 days). Next service window: 2026-04-22."
}
```

---

## Makefile Targets

| Target         | Description                                      |
|----------------|--------------------------------------------------|
| `install`      | `pip install -e ".[dev]"` + pre-commit hooks    |
| `download-data`| Print CMAPSS download instructions               |
| `train`        | Train all models, log to MLflow                 |
| `evaluate`     | Print Track A \| Track B comparison table       |
| `serve`        | Start FastAPI on port 8000                       |
| `benchmark`    | p50/p95/p99 latency (research vs deployment)     |
| `clean`        | Remove `__pycache__` and `.pyc` files            |
| `drift-report` | evidently HTML report + drift_summary.json       |
| `lint`         | black + ruff check                               |
| `test`         | pytest tests/                                    |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data validation | pandera |
| Feature engineering | pandas + scikit-learn |
| Classical ML | scikit-learn, XGBoost, Optuna |
| Deep learning | PyTorch (LSTM) |
| Survival analysis | lifelines (CoxPH) |
| Tracking + Registry | MLflow (DB-backed; Postgres) |
| ONNX (export only) | skl2onnx — sklearn models exported for fidelity testing; not used at serve time |
| Serving | FastAPI + Uvicorn |
| Monitoring | evidently (DataDriftPreset) |
| Containerisation | Docker Compose |
| Build automation | Makefile |

---

**Why SHAP for explainability?**  
XGBoost's native `pred_contribs=True` computes exact Shapley values (same C++ TreeShap algorithm as the `shap` library) — no extra dependency, no sampling, deterministic results. At batch_size=1 the overhead is ~0.5 ms. The `/explain` endpoint returns top-N feature contributions in log-odds space, allowing operators to understand *why* a high-risk prediction was made (e.g. sensor_11 HPC static pressure drift) before issuing a maintenance order.

---

*Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund — "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08.*
