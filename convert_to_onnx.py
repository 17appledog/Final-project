"""
convert_to_onnx.py
------------------
Run this ONCE locally to convert the XGBoost model to ONNX format.
The ONNX model is much smaller to deploy than the full xgboost package.
"""
import json
import numpy as np
from xgboost import XGBRegressor

with open("models/feature_names.json") as f:
    feature_names = json.load(f)

n_features = len(feature_names)
print(f"[INFO] Number of features: {n_features}")

# Load the model
model = XGBRegressor()
model.load_model("models/xgb_model.json")
print("[INFO] Model loaded.")

# Convert to ONNX using XGBoost's native export (requires 'onnx' package)
try:
    from xgboost import XGBRegressor
    model.save_model("models/xgb_model.onnx")
    print("[OK] Saved via XGBoost native ONNX export.")
except Exception as e:
    print(f"[WARN] Native export failed: {e}, trying onnxmltools...")
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools.convert import convert_xgboost

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    with open("models/xgb_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("[OK] Saved via onnxmltools.")

import os
size_mb = os.path.getsize("models/xgb_model.onnx") / (1024 * 1024)
print(f"[OK] ONNX model saved: models/xgb_model.onnx ({size_mb:.1f} MB)")
