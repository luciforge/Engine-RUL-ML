"""Run end-to-end inference and print per-engine predictions with uncertainty bounds."""
import sys, json
import numpy as np

sys.path.insert(0, ".")

from xgboost import XGBClassifier, XGBRegressor
from data.loader import load_fd
from features.pipeline import get_feature_cols
from labels.rul import add_rul
from labels.binary import add_binary_label
from models.classical.rf_xgb import load_conformal_artifact
from policy.notification import evaluate_alert, CostPolicy

# ── Load models ──────────────────────────────────────────────────────────────
clf = XGBClassifier(); clf.load_model("artifacts/best_model.json")
reg = XGBRegressor(); reg.load_model("artifacts/rul_regressor.json")
rul_lo = XGBRegressor(); rul_lo.load_model("artifacts/rul_lower.json")
rul_hi = XGBRegressor(); rul_hi.load_model("artifacts/rul_upper.json")
conf = load_conformal_artifact("artifacts/conformal_qhat.json")
qhat = conf["q_hat"]

# ── Load test data ────────────────────────────────────────────────────────────
test = load_fd("FD001", split="test")
feature_cols = get_feature_cols(test)
test = add_rul(test)
test = add_binary_label(test)

# ── Per-engine last-cycle inference ──────────────────────────────────────────
print()
print("=" * 80)
print("  UNCERTAINTY-AWARE PREDICTIONS  |  FD001 test set — last observed cycle")
print("=" * 80)
header = f"{'Eng':>4}  {'RUL_true':>8}  {'Risk':>6}  {'Risk_lo':>7}  {'Risk_hi':>7}  {'RUL_pred':>8}  {'RUL_lo':>6}  {'RUL_hi':>6}  Action"
print(header)
print("-" * 90)

for uid in sorted(test["unit_id"].unique())[:10]:
    eng = test[test["unit_id"] == uid].sort_values("cycle")
    row = eng.iloc[[-1]]
    arr = row[feature_cols].values.astype("float32")

    risk = float(clf.predict_proba(arr)[:, 1][0])
    rul_pred = float(reg.predict(arr)[0])
    rul_true = float(row["rul"].iloc[0])
    rul_lower = float(rul_lo.predict(arr)[0])
    rul_upper = float(rul_hi.predict(arr)[0])
    risk_lower = max(0.0, risk - qhat)
    risk_upper = min(1.0, risk + qhat)

    alert = evaluate_alert(int(uid), risk, rul_pred)
    action_short = alert.action[:16] if alert.action else "continue"

    print(
        f"{uid:>4}  {rul_true:>8.0f}  {risk:>6.3f}  {risk_lower:>7.3f}  {risk_upper:>7.3f}"
        f"  {rul_pred:>8.1f}  {rul_lower:>6.1f}  {rul_upper:>6.1f}  {action_short}"
    )

# ── Model comparison table ────────────────────────────────────────────────────
print()
print("=" * 80)
print("  MODEL COMPARISON  |  FD001 (XGBoost deployed, no-HPO run)")
print("=" * 80)
print(f"{'Model':<35}  {'PR-AUC':>7}  {'F1':>6}  {'Brier':>7}  {'ECE':>7}  {'Note'}")
print("-" * 80)

models = [
    ("XGBoost (deployment)", 0.9802, 0.9165, 0.0201, 0.0167, "best_model.json"),
    ("Random Forest",        0.9803, 0.9238, 0.0181, 0.0171, "classical"),
    ("Logistic Regression",  0.9759, 0.8924, 0.0251, 0.0215, "classical"),
    ("Threshold baseline",   0.1511, 0.0000, 0.1511, 0.1511, "na-ive"),
    ("CoxPH survival",        None,   None,   None,   None,   "C-index=0.7245"),
    ("RUL regressor (XGB)",   None,   None,   None,   None,   "MAE=68.7 RMSE=81.0"),
]
for name, prauc, f1, brier, ece, note in models:
    pa = f"{prauc:.4f}" if prauc is not None else "  —    "
    f1s = f"{f1:.4f}" if f1 is not None else "  —   "
    bs = f"{brier:.4f}" if brier is not None else "  —    "
    es = f"{ece:.4f}" if ece is not None else "  —    "
    print(f"{name:<35}  {pa:>7}  {f1s:>6}  {bs:>7}  {es:>7}  {note}")

# ── Calibrated uncertainty summary ───────────────────────────────────────────
print()
print("=" * 80)
print("  UNCERTAINTY CALIBRATION SUMMARY")
print("=" * 80)
print(f"  Conformal coverage target : {1 - conf['alpha']:.0%}")
print(f"  Conformal q_hat           : {qhat:.6f}")
print(f"  Quantile RUL q=0.10 model : artifacts/rul_lower.json")
print(f"  Quantile RUL q=0.90 model : artifacts/rul_upper.json")

# ── Cost-policy single-engine demo ───────────────────────────────────────────
print()
print("=" * 80)
print("  COST-SENSITIVE POLICY DEMO  (engine 1, last cycle)")
print("=" * 80)
eng1 = test[test["unit_id"] == 1].sort_values("cycle")
arr1 = eng1.iloc[[-1]][feature_cols].values.astype("float32")
risk1 = float(clf.predict_proba(arr1)[:, 1][0])
rul1 = float(reg.predict(arr1)[0])
alert1 = evaluate_alert(1, risk1, rul1)
policy = CostPolicy.from_config()
print(f"  Risk score     : {risk1:.4f}")
print(f"  Predicted RUL  : {rul1:.1f} cycles")
print(f"  Urgency        : {alert1.urgency}")
print(f"  Recommended action : {alert1.action}")
print(f"  Expected cost  : EUR {alert1.expected_cost:,.2f}")
print(f"  Service date   : {alert1.recommended_service_date}")
print()
print(f"  Cost parameters (from config.yaml):")
print(f"    Planned replacement  : EUR {policy.cost_replacement:,.0f}")
print(f"    Unplanned failure    : EUR {policy.cost_unplanned_failure:,.0f}")
print(f"    Inspection           : EUR {policy.cost_inspection:,.0f}")
print(f"    False alarm          : EUR {policy.cost_false_alarm:,.0f}")

print()
print("=" * 80)
print("  ALL ARTIFACTS")
print("=" * 80)
import os
for fname in ["best_model.json", "rul_regressor.json", "conformal_qhat.json",
              "rul_lower.json", "rul_upper.json"]:
    path = f"artifacts/{fname}"
    size = os.path.getsize(path)
    print(f"  {fname:<30}  {size:>10,} bytes")
