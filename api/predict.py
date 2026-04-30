"""
api/predict.py
--------------
Pure Python serverless function for Vercel using ONNX.
"""
import os
import json
import numpy as np
import onnxruntime as ort
from http.server import BaseHTTPRequestHandler

# ──────────────────────────────────────────────
# Load artefacts once at cold-start
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def _load():
    onnx_path = os.path.join(MODELS_DIR, "xgb_model.onnx")
    
    # Load ONNX session
    # Note: onnxruntime is much lighter than xgboost
    session = ort.InferenceSession(onnx_path)

    with open(os.path.join(MODELS_DIR, "scaler.json")) as f:
        scaler = json.load(f)
    
    with open(os.path.join(MODELS_DIR, "label_encoders.json")) as f:
        les = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        meta = json.load(f)
        
    return session, scaler, les, feat_names, meta

LOAD_ERROR = None
try:
    print("[INFO] Loading ONNX model...")
    SESSION, SCALER, LABEL_ENCODERS, FEATURE_NAMES, META = _load()
    DEFAULT_VALUES = META["default_values"]
    SKEW_COLS      = META["skew_cols"]
    print("[OK] ONNX model loaded successfully.")
except Exception as exc:
    LOAD_ERROR = str(exc)
    print(f"[ERROR] Could not load models: {exc}")
    SESSION = SCALER = LABEL_ENCODERS = FEATURE_NAMES = DEFAULT_VALUES = SKEW_COLS = None

def _build_row(feat: dict) -> np.ndarray:
    row = dict(DEFAULT_VALUES)

    row["OverallQual"]   = feat.get("OverallQual", 5)
    row["GrLivArea"]     = feat.get("GrLivArea", 1500.0)
    row["YearBuilt"]     = feat.get("YearBuilt", 1970)
    row["GarageCars"]    = feat.get("GarageCars", 2)
    row["KitchenAbvGrd"] = feat.get("KitchenAbvGrd", 1)
    row["Fireplaces"]    = feat.get("Fireplaces", 0)
    row["FullBath"]      = feat.get("FullBath", 2)
    row["LotArea"]       = feat.get("LotArea", 10000.0)
    row["TotRmsAbvGrd"]  = feat.get("TotRmsAbvGrd", 6)

    nbhd_raw = feat.get("Neighborhood", "")
    if "Neighborhood" in LABEL_ENCODERS:
        classes = LABEL_ENCODERS["Neighborhood"]
        if nbhd_raw in classes:
            row["Neighborhood"] = classes.index(nbhd_raw)
        else:
            row["Neighborhood"] = classes.index(classes[0])
    else:
        row["Neighborhood"] = 0

    yr_sold = int(row.get("YrSold", 2010))
    row["YearBuilt"]      = feat.get("YearBuilt", 1970)
    row["YearRemodAdd"]   = row.get("YearRemodAdd", row["YearBuilt"])

    total_bsmt = row.get("TotalBsmtSF", 0)
    first_flr  = row.get("1stFlrSF", 0)
    second_flr = row.get("2ndFlrSF", 0)

    row["TotalSF"]      = feat.get("TotalSF", 1500.0)
    row["TotalBath"]    = (feat.get("FullBath", 2) + 0.5 * row.get("HalfBath", 0) +
                           row.get("BsmtFullBath", 0) + 0.5 * row.get("BsmtHalfBath", 0))
    row["TotalPorchSF"] = (row.get("OpenPorchSF", 0) + row.get("EnclosedPorch", 0) +
                           row.get("3SsnPorch", 0) + row.get("ScreenPorch", 0))
    row["HouseAge"]     = feat.get("HouseAge", 20)
    row["RemodelAge"]   = yr_sold - int(row.get("YearRemodAdd", row["YearBuilt"]))
    row["HasRemodel"]   = int(row.get("YearRemodAdd", row["YearBuilt"]) != row["YearBuilt"])
    row["HasGarage"]    = int(row.get("GarageArea", 0) > 0)
    row["HasBsmt"]      = int(total_bsmt > 0)
    row["HasFireplace"] = int(feat.get("Fireplaces", 0) > 0)
    row["Has2ndFloor"]  = int(second_flr > 0)
    row["HasWoodDeck"]  = int(row.get("WoodDeckSF", 0) > 0)
    row["HasOpenPorch"] = int(row.get("OpenPorchSF", 0) > 0)
    row["HasPool"]      = int(row.get("PoolArea", 0) > 0)
    row["LotPerSF"]     = feat.get("LotArea", 10000.0) / (feat.get("TotalSF", 1500.0) + 1)
    row["GarageRatio"]  = row.get("GarageArea", 0) / (feat.get("TotalSF", 1500.0) + 1)
    row["OverallQual2"] = feat.get("OverallQual", 5) ** 2
    row["OverallQual3"] = feat.get("OverallQual", 5) ** 3
    row["QualPerSF"]    = feat.get("OverallQual", 5) / (feat.get("TotalSF", 1500.0) + 1)
    row["YearBuilt2"]   = feat.get("YearBuilt", 1970) ** 2
    row["TotalRooms"]   = feat.get("TotRmsAbvGrd", 6) + feat.get("FullBath", 2) + row.get("HalfBath", 0)
    row["AvgRoomSize"]  = feat.get("GrLivArea", 1500.0) / (feat.get("TotRmsAbvGrd", 6) + 1)

    row_values = [row.get(col, 0) for col in FEATURE_NAMES]
    return np.array([row_values], dtype=np.float32)

def _preprocess(X: np.ndarray) -> np.ndarray:
    skew_indices = [FEATURE_NAMES.index(col) for col in SKEW_COLS if col in FEATURE_NAMES]
    for idx in skew_indices:
        X[0, idx] = np.log1p(max(0, X[0, idx]))
        
    center = np.array(SCALER["center"])
    scale = np.array(SCALER["scale"])
    scale[scale == 0] = 1.0
    return (X - center) / scale


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        res = {"status": "ok", "model_loaded": SESSION is not None, "error": LOAD_ERROR}
        self.wfile.write(json.dumps(res).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            feat = json.loads(post_data.decode('utf-8'))
        except:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"detail": "Invalid JSON"}')
            return

        if SESSION is None:
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            err_msg = f"Model not loaded. Error: {LOAD_ERROR}"
            self.wfile.write(json.dumps({"detail": err_msg}).encode('utf-8'))
            return

        try:
            df_row = _build_row(feat)
            X = _preprocess(df_row)
            X = X.astype(np.float32)
            
            # ONNX Inference
            input_name = SESSION.get_inputs()[0].name
            output = SESSION.run(None, {input_name: X})
            log_pred = float(output[0][0])
            
            price = float(np.expm1(log_pred))
            price = max(50_000, min(price, 1_500_000))
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"predicted_price": round(price, 2)}).encode('utf-8'))
        except Exception as exc:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"detail": str(exc)}).encode('utf-8'))
