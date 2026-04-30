"""
convert_to_onnx.py
------------------
Correctly convert XGBoost to ONNX using onnxmltools.
"""
import json
import numpy as np
import os
from xgboost import XGBRegressor
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType

# Load metadata to get feature count
with open("models/feature_names.json") as f:
    feature_names = json.load(f)
n_features = len(feature_names)
print(f"[INFO] Number of features: {n_features}")

# Load the model
model = XGBRegressor()
model.load_model("models/xgb_model.json")
print("[INFO] Model loaded from JSON.")

# Convert to ONNX
initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

# Save the ONNX model
onnx_path = "models/xgb_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

size_kb = os.path.getsize(onnx_path) / 1024
print(f"[OK] ONNX model saved: {onnx_path} ({size_kb:.1f} KB)")
