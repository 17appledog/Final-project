import joblib
import json
import os
import numpy as np

MODELS_DIR = "models"

# 1. Scaler
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
scaler_data = {
    "center": scaler.center_.tolist() if scaler.center_ is not None else [],
    "scale": scaler.scale_.tolist() if scaler.scale_ is not None else []
}
with open(os.path.join(MODELS_DIR, "scaler.json"), "w") as f:
    json.dump(scaler_data, f)

# 2. Label Encoders
les = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
les_data = {}
for col, le in les.items():
    les_data[col] = list(le.classes_)
with open(os.path.join(MODELS_DIR, "label_encoders.json"), "w") as f:
    json.dump(les_data, f)

# 3. Feature names & Meta (already light, but let's convert to JSON to avoid joblib)
feat_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
    json.dump(list(feat_names), f)

meta = joblib.load(os.path.join(MODELS_DIR, "default_values.pkl"))
# handle np types in meta
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, default=convert)

print("Extraction complete.")
