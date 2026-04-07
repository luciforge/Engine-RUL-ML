"""Compute RMSE against the ground-truth RUL file for the deployment regressor."""
import numpy as np, sys
sys.path.insert(0, ".")
from data.loader import load_fd
from features.pipeline import get_feature_cols
from xgboost import XGBRegressor

test = load_fd("FD001", split="test")
feature_cols = get_feature_cols(test)
true_rul = np.loadtxt("../CMAPSSData/RUL_FD001.txt")

reg = XGBRegressor()
reg.load_model("artifacts/rul_regressor.json")

preds = []
for uid in sorted(test["unit_id"].unique()):
    eng = test[test["unit_id"] == uid].sort_values("cycle")
    arr = eng.iloc[[-1]][feature_cols].values.astype("float32")
    preds.append(float(reg.predict(arr)[0]))

preds = np.array(preds)
errors = preds - true_rul
rmse = float(np.sqrt(np.mean(errors**2)))
mae = float(np.mean(np.abs(errors)))

preds_c = np.clip(preds, 0, 125)
true_c = np.clip(true_rul, 0, 125)
rmse_c = float(np.sqrt(np.mean((preds_c - true_c)**2)))

print(f"Engines: {len(true_rul)}")
print(f"True RUL range: {true_rul.min():.0f}-{true_rul.max():.0f} mean={true_rul.mean():.1f}")
print(f"Pred RUL range: {preds.min():.0f}-{preds.max():.0f} mean={preds.mean():.1f}")
print(f"MAE  (raw):      {mae:.2f} cycles")
print(f"RMSE (raw):      {rmse:.2f} cycles")
print(f"RMSE (cap 125):  {rmse_c:.2f} cycles  (RUL capped at 125 per benchmark protocol)")
