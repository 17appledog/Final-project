import sys, os, json
import numpy as np
import xgboost as xgb
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Register XGBoost converter
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.common.shape_calculator import calculate_linear_regressor_output_shapes

update_registered_converter(
    xgb.XGBRegressor, 'XGBRegressor',
    calculate_linear_regressor_output_shapes, convert_xgboost
)

def log(msg):
    print(msg)
    sys.stdout.flush()

try:
    log("Starting ONNX conversion (skl2onnx) …")
    
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    n_features = len(feature_names)
    log(f"Features: {n_features}")
    
    model = xgb.XGBRegressor()
    model.load_model("models/xgb_model.json")
    log("Model loaded from JSON")
    
    # Convert using skl2onnx directly – much more stable
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset={'': 12, 'ai.onnx.ml': 3}
    )
    log("Conversion successful")
    
    onnx_path = "models/xgb_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    log(f"ONNX model saved to {onnx_path}")
    
    # Quick test
    dummy = np.zeros((1, n_features), dtype=np.float32)
    session = ort.InferenceSession(onnx_path)
    out = session.run(None, {session.get_inputs()[0].name: dummy})
    log(f"Test output: {out}")
    
except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
