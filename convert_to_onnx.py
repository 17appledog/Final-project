import sys
import os
import json

def log(msg):
    print(msg)
    sys.stdout.flush()

try:
    log("Starting conversion script...")
    
    import numpy as np
    import xgboost as xgb
    import onnxmltools
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    
    log("Imports successful")
    
    if not os.path.exists("models/feature_names.json"):
        log("ERROR: models/feature_names.json not found")
        sys.exit(1)
        
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    n_features = len(feature_names)
    log(f"Features found: {n_features}")
    
    if not os.path.exists("models/xgb_model.json"):
        log("ERROR: models/xgb_model.json not found")
        sys.exit(1)
        
    model = xgb.XGBRegressor()
    model.load_model("models/xgb_model.json")
    log("Model loaded from JSON")
    
    log("Starting ONNX conversion...")
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type, target_opset=12)
    log("Conversion successful")
    
    onnx_path = "models/xgb_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    log(f"ONNX model saved to {onnx_path}")
    
except Exception as e:
    log(f"CRITICAL ERROR: {str(e)}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)
