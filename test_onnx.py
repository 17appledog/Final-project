import onnxruntime as ort
import numpy as np
import json

session = ort.InferenceSession("models/xgb_model.onnx")
input_name = session.get_inputs()[0].name
print("Input name:", input_name)
print("Input shape:", session.get_inputs()[0].shape)

# Create dummy input based on feature names
with open("models/feature_names.json") as f:
    feat_names = json.load(f)

dummy_X = np.zeros((1, len(feat_names)), dtype=np.float32)

output = session.run(None, {input_name: dummy_X})
print("Output:", output)
print("Output shape:", output[0].shape)
