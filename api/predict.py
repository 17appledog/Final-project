import sys
import os
import json
import numpy as np
import onnxruntime as ort

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# 1. Load all lightweight configs at import time
with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

with open(os.path.join(MODEL_DIR, "scaler.json")) as f:
    sc = json.load(f)
    SCALER_CENTER = np.array(sc["center"], dtype=np.float32)
    SCALER_SCALE  = np.array(sc["scale"], dtype=np.float32)

with open(os.path.join(MODEL_DIR, "label_encoders.json")) as f:
    LABEL_ENCODERS = json.load(f)

with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)
    DEFAULT_VALUES = meta["default_values"]
    SKEW_COLS = meta.get("skew_cols", [])

# 2. Load the ONNX model (only once)
ONNX_PATH = os.path.join(MODEL_DIR, "xgb_model.onnx")
session = ort.InferenceSession(ONNX_PATH)
INPUT_NAME = session.get_inputs()[0].name
print("✅ Model ready", file=sys.stderr)

# -------------------------------------------------------------------
# Preprocessing (mirrors train_and_save.py)
# -------------------------------------------------------------------
def preprocess(input_json: dict) -> np.ndarray:
    features = DEFAULT_VALUES.copy()

    # 1. override with user input
    user = {
        "OverallQual": int(input_json["OverallQual"]),
        "GrLivArea": float(input_json["GrLivArea"]),
        "Neighborhood": input_json["Neighborhood"],
        "YearBuilt": int(input_json["YearBuilt"]),
        "TotalSF": float(input_json["TotalSF"]),
        "HouseAge": int(input_json["HouseAge"]),
        "GarageCars": int(input_json["GarageCars"]),
        "KitchenAbvGr": int(input_json["KitchenAbvGr"]),
        "Fireplaces": int(input_json["Fireplaces"]),
        "FullBath": int(input_json["FullBath"]),
        "LotArea": float(input_json["LotArea"]),
        "TotRmsAbvGrd": int(input_json["TotRmsAbvGrd"]),
    }
    for k, v in user.items():
        features[k] = v

    # 2. engineered features
    features["TotalBath"] = (features["FullBath"] + 0.5 * features.get("HalfBath", 0) +
                             features.get("BsmtFullBath", 0) + 0.5 * features.get("BsmtHalfBath", 0))
    features["TotalPorchSF"] = (features.get("OpenPorchSF", 0) + features.get("EnclosedPorch", 0) +
                                features.get("3SsnPorch", 0) + features.get("ScreenPorch", 0))
    features["HouseAge"] = features["YrSold"] - features["YearBuilt"]
    features["RemodelAge"] = features["YrSold"] - features.get("YearRemodAdd", features["YearBuilt"])
    features["HasRemodel"] = 1 if features.get("YearRemodAdd", features["YearBuilt"]) != features["YearBuilt"] else 0
    features["HasGarage"] = 1 if features.get("GarageArea", 0) > 0 else 0
    features["HasBsmt"] = 1 if features.get("TotalBsmtSF", 0) > 0 else 0
    features["HasFireplace"] = 1 if features["Fireplaces"] > 0 else 0
    features["Has2ndFloor"] = 1 if features.get("2ndFlrSF", 0) > 0 else 0
    features["HasWoodDeck"] = 1 if features.get("WoodDeckSF", 0) > 0 else 0
    features["HasOpenPorch"] = 1 if features.get("OpenPorchSF", 0) > 0 else 0
    features["HasPool"] = 1 if features.get("PoolArea", 0) > 0 else 0
    features["LotPerSF"] = features["LotArea"] / (features["TotalSF"] + 1)
    features["GarageRatio"] = features.get("GarageArea", 0) / (features["TotalSF"] + 1)
    features["OverallQual2"] = features["OverallQual"] ** 2
    features["OverallQual3"] = features["OverallQual"] ** 3
    features["QualPerSF"] = features["OverallQual"] / (features["TotalSF"] + 1)
    features["YearBuilt2"] = features["YearBuilt"] ** 2
    features["TotalRooms"] = features["TotRmsAbvGrd"] + features["FullBath"] + features.get("HalfBath", 0)
    features["AvgRoomSize"] = features["GrLivArea"] / (features["TotRmsAbvGrd"] + 1)

    # 3. label encode categoricals
    for col, classes in LABEL_ENCODERS.items():
        if col in features:
            val = features[col]
            if isinstance(val, str):
                idx = classes.index(val) if val in classes else 0
                features[col] = idx
            else:
                features[col] = int(float(val))

    # 4. build array in exact order of feature_names
    X = np.array([features[col] for col in FEATURE_NAMES], dtype=np.float32)

    # 5. log1p skewed features
    for col in SKEW_COLS:
        idx = FEATURE_NAMES.index(col)
        X[idx] = np.log1p(max(0, X[idx]))

    # 6. robust scaling (identical to training)
    X = (X - SCALER_CENTER) / (SCALER_SCALE + 1e-8)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X.reshape(1, -1)

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    OverallQual: int
    GrLivArea: float
    Neighborhood: str
    YearBuilt: int
    TotalSF: float
    HouseAge: int
    GarageCars: int
    KitchenAbvGr: int
    Fireplaces: int
    FullBath: int
    LotArea: float
    TotRmsAbvGrd: int

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/api/predict")
async def predict(request: PredictRequest):
    try:
        X = preprocess(request.dict())
        out = session.run(None, {INPUT_NAME: X})
        pred_log = float(out[0][0])
        price = np.expm1(pred_log)
        return {"predicted_price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
