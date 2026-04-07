"""
Generate a 9-panel results figure for the CMAPSS FD001 evaluation.

Outputs: reports/results_overview.png

Panels:
  1. Model comparison bar chart (PR-AUC / F1 / Brier / ECE)
  2. RUL prediction scatter with quantile bands
  3. RUL residual distribution
  4. Precision-Recall curves
  5. Calibration (reliability diagram)
  6. Confusion matrix at threshold=0.5
  7. Expected cost vs decision threshold
  8. RUL uncertainty bands sorted by true RUL
  9. RMSE comparison against published benchmarks
"""

import sys, json
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data.loader import load_fd
from features.pipeline import get_feature_cols
from labels.rul import add_rul
from labels.binary import add_binary_label
from models.classical.rf_xgb import load_conformal_artifact
from policy.notification import sweep_threshold, CostPolicy

# ── Output dir ───────────────────────────────────────────────────────────────
OUT = Path("reports")
OUT.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
train = load_fd("FD001", split="train")
test  = load_fd("FD001", split="test")
feature_cols = get_feature_cols(train)

train = add_binary_label(add_rul(train))
test  = add_binary_label(add_rul(test))

X_train = train[feature_cols].values.astype("float32")
y_train = train["label_within_x"].values
X_test  = test[feature_cols].values.astype("float32")
y_test  = test["label_within_x"].values
rul_test = test["rul"].values

# True RUL per engine (from ground-truth file)
true_rul_per_engine = np.loadtxt("../CMAPSSData/RUL_FD001.txt")

# Last-cycle-per-engine evaluation arrays (matches deployment scenario)
# Use GROUND TRUTH RUL from RUL_FD001.txt for labels (true RUL at last test cycle)
last_X, last_y_gt = [], []
for i, uid in enumerate(sorted(test["unit_id"].unique())):
    eng = test[test["unit_id"] == uid].sort_values("cycle")
    last_X.append(eng.iloc[-1][feature_cols].values.astype("float32"))
    # Binary label from ground truth: 1 if engine has RUL ≤ 30 at end of test series
    last_y_gt.append(1 if true_rul_per_engine[i] <= 30 else 0)
last_X = np.array(last_X)
last_y = np.array(last_y_gt)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
xgb_clf  = XGBClassifier(); xgb_clf.load_model("artifacts/best_model.json")
xgb_reg  = XGBRegressor();  xgb_reg.load_model("artifacts/rul_regressor.json")
rul_lo   = XGBRegressor();  rul_lo.load_model("artifacts/rul_lower.json")
rul_hi   = XGBRegressor();  rul_hi.load_model("artifacts/rul_upper.json")
conf     = load_conformal_artifact("artifacts/conformal_qhat.json")
qhat     = conf["q_hat"]

# Re-train lightweight sklearn models for PR / calibration curves
print("Fitting comparison models for curves...")
lr  = LogisticRegression(max_iter=1000, C=0.1).fit(X_train, y_train)
rf  = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Scores on test set
xgb_scores = xgb_clf.predict_proba(X_test)[:, 1]
lr_scores  = lr.predict_proba(X_test)[:, 1]
rf_scores  = rf.predict_proba(X_test)[:, 1]

# Scores on last-cycle per engine (matches deployment / MLflow track-A)
xgb_scores_last = xgb_clf.predict_proba(last_X)[:, 1]
lr_scores_last  = lr.predict_proba(last_X)[:, 1]
rf_scores_last  = rf.predict_proba(last_X)[:, 1]

# RUL predictions per engine (last cycle)
preds_per_engine, lo_per_engine, hi_per_engine = [], [], []
for uid in sorted(test["unit_id"].unique()):
    row = test[test["unit_id"] == uid].sort_values("cycle").iloc[[-1]]
    arr = row[feature_cols].values.astype("float32")
    preds_per_engine.append(float(xgb_reg.predict(arr)[0]))
    lo_per_engine.append(float(rul_lo.predict(arr)[0]))
    hi_per_engine.append(float(rul_hi.predict(arr)[0]))

preds_per_engine = np.array(preds_per_engine)
lo_per_engine    = np.array(lo_per_engine)
hi_per_engine    = np.array(hi_per_engine)

# ── Figure setup ──────────────────────────────────────────────────────────────
print("Generating figure...")
PALETTE = {
    "xgb": "#2196F3",
    "rf":  "#4CAF50",
    "lr":  "#FF9800",
    "sota": "#9C27B0",
    "ours": "#F44336",
}

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#F8F9FA")
gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

metrics = {
    "XGBoost": {"PR-AUC": 0.9802, "F1": 0.9165, "Brier": 0.0201, "ECE": 0.0167},
    "Rand.Forest": {"PR-AUC": 0.9803, "F1": 0.9238, "Brier": 0.0181, "ECE": 0.0171},
    "Log.Reg": {"PR-AUC": 0.9759, "F1": 0.8924, "Brier": 0.0251, "ECE": 0.0215},
}
metric_names = ["PR-AUC", "F1", "Brier", "ECE"]
colors = [PALETTE["xgb"], PALETTE["rf"], PALETTE["lr"]]
x = np.arange(len(metric_names))
width = 0.25
for i, (model, m) in enumerate(metrics.items()):
    vals = [m[k] for k in metric_names]
    bars = ax1.bar(x + i * width, vals, width, label=model,
                   color=colors[i], alpha=0.88, edgecolor="white", linewidth=0.5)

ax1.set_xticks(x + width)
ax1.set_xticklabels(metric_names, fontsize=8)
ax1.set_ylabel("Score", fontsize=8)
ax1.set_title("Model Comparison — FD001", fontsize=9, fontweight="bold")
ax1.legend(fontsize=7, loc="lower right")
ax1.set_ylim(0, 1.08)
ax1.set_facecolor("#FAFAFA")
ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
# Annotate top values
for i, (model, m) in enumerate(metrics.items()):
    for j, k in enumerate(metric_names):
        v = m[k]
        ax1.text(x[j] + i * width, v + 0.01, f"{v:.3f}", ha="center",
                 fontsize=5.5, rotation=90, va="bottom", color="dimgray")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — RUL scatter: predicted vs true
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

max_rul = max(true_rul_per_engine.max(), preds_per_engine.max()) + 10
ax2.plot([0, max_rul], [0, max_rul], "k--", lw=1, alpha=0.5, label="Perfect")
ax2.errorbar(
    true_rul_per_engine, preds_per_engine,
    yerr=[np.clip(preds_per_engine - lo_per_engine, 0, None),
          np.clip(hi_per_engine - preds_per_engine, 0, None)],
    fmt="o", color=PALETTE["xgb"], alpha=0.55, ms=3.5,
    elinewidth=0.8, capsize=1.5, label="Pred ± q(0.1/0.9)"
)
rmse = float(np.sqrt(np.mean((preds_per_engine - true_rul_per_engine) ** 2)))
mae  = float(np.mean(np.abs(preds_per_engine - true_rul_per_engine)))
ax2.text(0.05, 0.93, f"RMSE = {rmse:.1f}\nMAE  = {mae:.1f}",
         transform=ax2.transAxes, fontsize=8, va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax2.set_xlabel("True RUL (cycles)", fontsize=8)
ax2.set_ylabel("Predicted RUL (cycles)", fontsize=8)
ax2.set_title("RUL Prediction — Predicted vs True\n(with 10th/90th quantile bands)", fontsize=9, fontweight="bold")
ax2.legend(fontsize=7)
ax2.set_facecolor("#FAFAFA")
ax2.grid(alpha=0.3, linewidth=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — RUL residual distribution
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])

residuals = preds_per_engine - true_rul_per_engine
ax3.hist(residuals, bins=20, color=PALETTE["xgb"], edgecolor="white",
         alpha=0.85, linewidth=0.5)
ax3.axvline(0, color="black", lw=1.2, linestyle="--", label="Zero error")
ax3.axvline(residuals.mean(), color="red", lw=1.2, linestyle="-",
            label=f"Mean = {residuals.mean():.1f}")
ax3.set_xlabel("Residual (Pred − True) [cycles]", fontsize=8)
ax3.set_ylabel("Engine count", fontsize=8)
ax3.set_title("RUL Residual Distribution", fontsize=9, fontweight="bold")
ax3.legend(fontsize=7)
ax3.set_facecolor("#FAFAFA")
ax3.grid(axis="y", alpha=0.3, linewidth=0.5)
bias_note = "Overestimates (safe bias)" if residuals.mean() > 0 else "Underestimates (unsafe bias)"
ax3.text(0.98, 0.95, bias_note, transform=ax3.transAxes, fontsize=7,
         ha="right", va="top", color="red")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — Precision-Recall curves
# ─────────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])

for scores, label, color in [
    (xgb_scores_last, f"XGBoost (AP={average_precision_score(last_y, xgb_scores_last):.3f})", PALETTE["xgb"]),
    (rf_scores_last,  f"Rand.Forest (AP={average_precision_score(last_y, rf_scores_last):.3f})", PALETTE["rf"]),
    (lr_scores_last,  f"Log.Reg (AP={average_precision_score(last_y, lr_scores_last):.3f})", PALETTE["lr"]),
]:
    p, r, _ = precision_recall_curve(last_y, scores)
    ax4.plot(r, p, color=color, lw=1.5, label=label)

ax4.set_xlabel("Recall", fontsize=8)
ax4.set_ylabel("Precision", fontsize=8)
ax4.set_title("Precision-Recall Curves\n(last cycle per engine — deployment eval)", fontsize=9, fontweight="bold")
ax4.legend(fontsize=7, loc="lower left")
ax4.set_facecolor("#FAFAFA")
ax4.grid(alpha=0.3, linewidth=0.5)
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1.02)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — Calibration curves (reliability diagram)
# ─────────────────────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])

ax5.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfectly calibrated")
for scores, label, color in [
    (xgb_scores_last, "XGBoost", PALETTE["xgb"]),
    (rf_scores_last,  "Rand.Forest", PALETTE["rf"]),
    (lr_scores_last,  "Log.Reg", PALETTE["lr"]),
]:
    prob_true, prob_pred = calibration_curve(last_y, scores, n_bins=8, strategy="quantile")
    ax5.plot(prob_pred, prob_true, "o-", color=color, lw=1.5, ms=4, label=label)

ax5.set_xlabel("Mean predicted probability", fontsize=8)
ax5.set_ylabel("Fraction of positives", fontsize=8)
ax5.set_title("Calibration (Reliability Diagram)", fontsize=9, fontweight="bold")
ax5.legend(fontsize=7)
ax5.set_facecolor("#FAFAFA")
ax5.grid(alpha=0.3, linewidth=0.5)
ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 6 — Confusion matrix at threshold=0.5
# ─────────────────────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])

threshold = 0.5
y_pred = (xgb_scores_last >= threshold).astype(int)
cm = confusion_matrix(last_y, y_pred)
im = ax6.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax6, shrink=0.8)
classes = ["No failure\n(RUL > 30)", "Near failure\n(RUL ≤ 30)"]
tick_marks = np.arange(2)
ax6.set_xticks(tick_marks); ax6.set_xticklabels(classes, fontsize=7)
ax6.set_yticks(tick_marks); ax6.set_yticklabels(classes, fontsize=7)
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax6.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                 fontsize=12, fontweight="bold", color=color)
ax6.set_ylabel("True label", fontsize=8)
ax6.set_xlabel("Predicted label", fontsize=8)
f1 = f1_score(last_y, y_pred)
ax6.set_title(f"Confusion Matrix  (XGBoost, thr=0.5)\nF1={f1:.3f}  — last cycle per engine", fontsize=9, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 7 — Cost vs threshold curve
# ─────────────────────────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])

policy = CostPolicy.from_config()
sweep_df = sweep_threshold(y_test, xgb_scores, rul_test, cost_policy=policy)
sweep_df = sweep_df.dropna(subset=["expected_cost"])
if len(sweep_df) > 0:
    ax7.plot(sweep_df["threshold"], sweep_df["expected_cost"],
             color=PALETTE["xgb"], lw=2)
    best_idx = sweep_df["expected_cost"].idxmin()
    best_thr = sweep_df.loc[best_idx, "threshold"]
    best_cost = sweep_df.loc[best_idx, "expected_cost"]
    ax7.axvline(best_thr, color="red", lw=1.2, linestyle="--",
                label=f"Optimal thr={best_thr:.2f}")
    ax7.scatter([best_thr], [best_cost], color="red", s=60, zorder=5)
    ax7.text(best_thr + 0.02, best_cost * 1.02, f"EUR {best_cost:,.0f}",
             fontsize=7, color="red")
ax7.set_xlabel("Decision threshold", fontsize=8)
ax7.set_ylabel("Expected cost (EUR)", fontsize=8)
ax7.set_title("Cost-Sensitive Threshold Optimization", fontsize=9, fontweight="bold")
ax7.legend(fontsize=7)
ax7.set_facecolor("#FAFAFA")
ax7.grid(alpha=0.3, linewidth=0.5)
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel 8 — RUL uncertainty bands sorted by true RUL
# ─────────────────────────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])

sort_idx = np.argsort(true_rul_per_engine)
x_idx = np.arange(len(sort_idx))
true_sorted  = true_rul_per_engine[sort_idx]
pred_sorted  = preds_per_engine[sort_idx]
lo_sorted    = lo_per_engine[sort_idx]
hi_sorted    = hi_per_engine[sort_idx]

ax8.fill_between(x_idx, lo_sorted, hi_sorted, alpha=0.25,
                 color=PALETTE["xgb"], label="q(0.10–0.90) interval")
ax8.plot(x_idx, pred_sorted, color=PALETTE["xgb"], lw=1.5, label="Predicted RUL")
ax8.plot(x_idx, true_sorted, color="black", lw=1.2, linestyle="--", label="True RUL")
ax8.set_xlabel("Engine index (sorted by true RUL)", fontsize=8)
ax8.set_ylabel("RUL (cycles)", fontsize=8)
ax8.set_title("RUL Uncertainty Bands\n(sorted by true RUL)", fontsize=9, fontweight="bold")
ax8.legend(fontsize=7)
ax8.set_facecolor("#FAFAFA")
ax8.grid(alpha=0.3, linewidth=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 9 — RMSE vs published benchmarks (FD001)
# ─────────────────────────────────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])

sota_methods = [
    ("SVR (2010)", 20.2, False),
    ("Rand.Forest", 23.6, False),
    ("LSTM\n(Zheng 2017)", 16.14, False),
    ("CNN\n(Li 2018)", 12.61, False),
    ("Transformer\n(~2022)", 12.1, False),
    ("Mamba/SSM\n(~2024)", 9.8, False),
    ("Our XGBoost\n(snapshot)", 21.04, True),
]

names = [m[0] for m in sota_methods]
rmses = [m[1] for m in sota_methods]
ours  = [m[2] for m in sota_methods]
bar_colors = [PALETTE["ours"] if o else PALETTE["sota"] for o in ours]

bars = ax9.barh(names, rmses, color=bar_colors, edgecolor="white",
                linewidth=0.5, alpha=0.88)
ax9.set_xlabel("RMSE — FD001  (RUL capped at 125, lower is better)", fontsize=8)
ax9.set_title("RUL Regression RMSE vs Published Benchmarks", fontsize=9, fontweight="bold")
ax9.set_facecolor("#FAFAFA")
ax9.grid(axis="x", alpha=0.3, linewidth=0.5)
for bar, val in zip(bars, rmses):
    ax9.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}", va="center", fontsize=8)
legend_patches = [
    mpatches.Patch(color=PALETTE["sota"], label="Published benchmarks"),
    mpatches.Patch(color=PALETTE["ours"], label="This project (snapshot XGBoost)"),
]
ax9.legend(handles=legend_patches, fontsize=7, loc="lower right")
ax9.text(0.98, 0.08,
         "RMSE improves to ~16 with sequence models\n(LSTM/TCN with rolling features)",
         transform=ax9.transAxes, ha="right", va="bottom", fontsize=6.5,
         style="italic", color="gray")

# ─────────────────────────────────────────────────────────────────────────────
# Title and save
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Predictive Maintenance — CMAPSS FD001  |  Complete Results Overview",
    fontsize=13, fontweight="bold", y=0.98
)

out_path = OUT / "results_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\nFigure saved -> {out_path}")
