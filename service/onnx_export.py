"""Export trained models to ONNX format and validate output fidelity.

Supported model types:
  - XGBoost / sklearn classifiers  → skl2onnx
  - PyTorch LSTM                    → torch.onnx.export

Validation: ONNX output must match native output within 1e-4 absolute tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_xgboost(
    model,
    feature_cols: list[str],
    output_path: Path,
) -> Path:
    """Export a fitted model to the appropriate format.

    XGBoost models (XGBClassifier / XGBRegressor) are saved in XGBoost's
    native JSON format. XGBoost 3.x removed ONNX export; JSON is the stable
    round-trip format and is loaded directly by the FastAPI service.
    Other sklearn estimators are exported to ONNX via skl2onnx (opset 17).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "get_booster"):
        # Save as .json regardless of the requested extension.
        json_path = output_path.with_suffix(".json")
        model.save_model(str(json_path))
        return json_path

    # Fallback: skl2onnx for LogisticRegression / RandomForest etc.
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    n_features = len(feature_cols)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    # zipmap=False: return probabilities as a plain ndarray [N, n_classes]
    # instead of a list of dicts, making output consistent across estimators.
    options = {type(model): {"zipmap": False}}
    onnx_model = convert_sklearn(
        model, initial_types=initial_type, target_opset=17, options=options
    )
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return output_path


def validate_onnx(
    native_fn,
    onnx_path: Path,
    X: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """Return True if max absolute deviation between native and ONNX outputs <= tol.

    Handles two output layouts:
      - Single output [N, 2] or [N, 1]: used when zipmap=False (classifiers) or regression.
      - Dual output (labels[N], proba[N, 2]): legacy skl2onnx without zipmap=False.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: X.astype(np.float32)})
    onnx_out = np.asarray(onnx_outputs[0])

    # skl2onnx classifiers emit (predicted_labels[N], probas[N, C]).
    # Detect by checking if first output has integer dtype.
    if len(onnx_outputs) >= 2 and np.issubdtype(onnx_out.dtype, np.integer):
        raw = onnx_outputs[1]
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            # ZipMap fallback
            classes = sorted(raw[0].keys())
            onnx_out = np.array([[d[c] for c in classes] for d in raw], dtype=np.float64)
        else:
            onnx_out = np.asarray(raw, dtype=np.float64)

    native_out = np.array(native_fn(X))
    # Extract positive-class probability column from [N, 2] arrays
    if native_out.ndim == 2 and native_out.shape[1] == 2:
        native_out = native_out[:, 1]
    else:
        native_out = native_out.ravel()
    if onnx_out.ndim == 2 and onnx_out.shape[1] == 2:
        onnx_out = onnx_out[:, 1]
    else:
        onnx_out = onnx_out.ravel()

    max_diff = float(np.abs(onnx_out - native_out).max())
    return max_diff <= tol
