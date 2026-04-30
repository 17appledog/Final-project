import sys
print("Starting...")
import json
print("json imported")
import numpy as np
print("numpy imported")
from xgboost import XGBRegressor
print("xgboost imported")

print("Loading metadata...")
with open("models/feature_names.json") as f:
    feature_names = json.load(f)
n_features = len(feature_names)
print(f"Number of features: {n_features}")

print("Loading model...")
model = XGBRegressor()
model.load_model("models/xgb_model.json")
print("Model loaded from JSON.")

print("Importing onnxmltools...")
try:
    import onnxmltools
    print("onnxmltools imported")
    from skl2onnx.common.data_types import FloatTensorType
    print("skl2onnx imported")
except Exception as e:
    print(f"Error importing onnx tools: {e}")
    sys.exit(1)

print("Converting...")
try:
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    print("Conversion done.")
    with open("models/xgb_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Saved successfully.")
except Exception as e:
    print(f"Error during conversion: {e}")
