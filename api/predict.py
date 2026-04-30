"""
api/predict.py
--------------
FastAPI serverless function for Vercel.
Endpoint: POST /api/predict
Body:     JSON with the 12 user-facing fields
Returns:  {"predicted_price": float}
"""

from __future__ import annotations

import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Bootstrap FastAPI
# ──────────────────────────────────────────────
app = FastAPI(title="House Price Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static files (index.html, style.css, script.js)
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

# Note: We mount this AT THE END to ensure /api routes take precedence

# ──────────────────────────────────────────────
# Load artefacts once at cold-start
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _load():
    from xgboost import XGBRegressor
    model_json = os.path.join(MODELS_DIR, "xgb_model.json")

    # Native format: faster load AND faster predict
    model = XGBRegressor()
    model.load_model(model_json)

    with open(os.path.join(MODELS_DIR, "scaler.json")) as f:
        scaler = json.load(f)
    
    with open(os.path.join(MODELS_DIR, "label_encoders.json")) as f:
        les = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "meta.json")) as f:
        meta = json.load(f)
        
    return model, scaler, les, feat_names, meta


try:
    print("[INFO] Loading model artefacts...")
    XGB_MODEL, SCALER, LABEL_ENCODERS, FEATURE_NAMES, META = _load()
    DEFAULT_VALUES = META["default_values"]
    SKEW_COLS      = META["skew_cols"]
    print("[OK] Models loaded successfully.")
except Exception as exc:
    print(f"[ERROR] Could not load models: {exc}")
    XGB_MODEL = SCALER = LABEL_ENCODERS = FEATURE_NAMES = DEFAULT_VALUES = SKEW_COLS = None

# ──────────────────────────────────────────────
# Neighborhood options (from Ames dataset)
# ──────────────────────────────────────────────
NEIGHBORHOODS = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes",   "NoRidge", "NPkVill", "NridgHt", "NWAmes",  "OldTown",
    "SWISU",   "Sawyer",  "SawyerW", "Somerst", "StoneBr", "Timber",
    "Veenker",
]

# ──────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────
class HouseFeatures(BaseModel):
    OverallQual:   int   = Field(..., ge=1, le=10,    description="Overall material & finish quality (1–10)")
    GrLivArea:     float = Field(..., ge=300, le=6000, description="Above grade living area (sq ft)")
    Neighborhood:  str   = Field(...,                  description="Physical location within Ames city limits")
    YearBuilt:     int   = Field(..., ge=1872, le=2010,description="Original construction year")
    TotalSF:       float = Field(..., ge=300, le=12000,description="Total square footage (basement + 1st + 2nd floor)")
    HouseAge:      int   = Field(..., ge=0, le=150,    description="Age of the house in years")
    GarageCars:    int   = Field(..., ge=0, le=4,      description="Size of garage in car capacity")
    KitchenAbvGrd: int   = Field(..., ge=0, le=3,      description="Kitchens above grade")
    Fireplaces:    int   = Field(..., ge=0, le=3,      description="Number of fireplaces")
    FullBath:      int   = Field(..., ge=0, le=4,      description="Full bathrooms above grade")
    LotArea:       float = Field(..., ge=1300, le=215000, description="Lot size (sq ft)")
    TotRmsAbvGrd:  int   = Field(..., ge=2, le=14,     description="Total rooms above grade (excl. bathrooms)")


# ──────────────────────────────────────────────
# Pre-processing helpers (mirror train_and_save.py)
# ──────────────────────────────────────────────
def _build_row(feat: HouseFeatures) -> np.ndarray:
    """
    Construct a 1-row numpy array with ALL columns expected by the model,
    starting from default values and overriding with the 12 user inputs.
    """
    row = dict(DEFAULT_VALUES)  # copy medians / modes

    # ── Override user-supplied fields ──────────────────────────────────────
    row["OverallQual"]   = feat.OverallQual
    row["GrLivArea"]     = feat.GrLivArea
    row["YearBuilt"]     = feat.YearBuilt
    row["GarageCars"]    = feat.GarageCars
    row["KitchenAbvGrd"] = feat.KitchenAbvGrd
    row["Fireplaces"]    = feat.Fireplaces
    row["FullBath"]      = feat.FullBath
    row["LotArea"]       = feat.LotArea
    row["TotRmsAbvGrd"]  = feat.TotRmsAbvGrd

    # Neighborhood – apply the same LabelEncoder logic used at training
    nbhd_raw = feat.Neighborhood
    if "Neighborhood" in LABEL_ENCODERS:
        classes = LABEL_ENCODERS["Neighborhood"]
        if nbhd_raw in classes:
            row["Neighborhood"] = classes.index(nbhd_raw)
        else:
            row["Neighborhood"] = classes.index(classes[0])
    else:
        row["Neighborhood"] = 0

    # Reconstruct dependent raw columns so engineered features are correct
    # We use the user-supplied TotalSF & HouseAge directly, but also need
    # the raw columns that feed into them.  Use defaults where not supplied.
    yr_sold = int(row.get("YrSold", 2010))
    row["YearBuilt"]      = feat.YearBuilt
    row["YearRemodAdd"]   = row.get("YearRemodAdd", feat.YearBuilt)

    # ── Re-create engineered features (same order as train_and_save.py) ──
    total_bsmt = row.get("TotalBsmtSF", 0)
    first_flr  = row.get("1stFlrSF", 0)
    second_flr = row.get("2ndFlrSF", 0)

    row["TotalSF"]      = feat.TotalSF  # user provided
    row["TotalBath"]    = (feat.FullBath + 0.5 * row.get("HalfBath", 0) +
                           row.get("BsmtFullBath", 0) + 0.5 * row.get("BsmtHalfBath", 0))
    row["TotalPorchSF"] = (row.get("OpenPorchSF", 0) + row.get("EnclosedPorch", 0) +
                           row.get("3SsnPorch", 0) + row.get("ScreenPorch", 0))
    row["HouseAge"]     = feat.HouseAge
    row["RemodelAge"]   = yr_sold - int(row.get("YearRemodAdd", feat.YearBuilt))
    row["HasRemodel"]   = int(row.get("YearRemodAdd", feat.YearBuilt) != feat.YearBuilt)
    row["HasGarage"]    = int(row.get("GarageArea", 0) > 0)
    row["HasBsmt"]      = int(total_bsmt > 0)
    row["HasFireplace"] = int(feat.Fireplaces > 0)
    row["Has2ndFloor"]  = int(second_flr > 0)
    row["HasWoodDeck"]  = int(row.get("WoodDeckSF", 0) > 0)
    row["HasOpenPorch"] = int(row.get("OpenPorchSF", 0) > 0)
    row["HasPool"]      = int(row.get("PoolArea", 0) > 0)
    row["LotPerSF"]     = feat.LotArea / (feat.TotalSF + 1)
    row["GarageRatio"]  = row.get("GarageArea", 0) / (feat.TotalSF + 1)
    row["OverallQual2"] = feat.OverallQual ** 2
    row["OverallQual3"] = feat.OverallQual ** 3
    row["QualPerSF"]    = feat.OverallQual / (feat.TotalSF + 1)
    row["YearBuilt2"]   = feat.YearBuilt ** 2
    row["TotalRooms"]   = feat.TotRmsAbvGrd + feat.FullBath + row.get("HalfBath", 0)
    row["AvgRoomSize"]  = feat.GrLivArea / (feat.TotRmsAbvGrd + 1)

    # Build numpy array with exact column order
    row_values = [row.get(col, 0) for col in FEATURE_NAMES]
    return np.array([row_values], dtype=np.float32)



def _preprocess(X: np.ndarray) -> np.ndarray:
    """Apply log1p + RobustScaler identical to training."""
    # Find indices for SKEW_COLS
    skew_indices = [FEATURE_NAMES.index(col) for col in SKEW_COLS if col in FEATURE_NAMES]
    
    # log1p transformation on SKEW_COLS
    for idx in skew_indices:
        X[0, idx] = np.log1p(max(0, X[0, idx]))
        
    # RobustScaler transformation
    # (X - center) / scale
    center = np.array(SCALER["center"])
    scale = np.array(SCALER["scale"])
    
    # Avoid division by zero
    scale[scale == 0] = 1.0
    
    return (X - center) / scale


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": XGB_MODEL is not None}


@app.get("/api/neighborhoods")
def neighborhoods():
    return {"neighborhoods": NEIGHBORHOODS}


@app.post("/api/predict")
def predict(feat: HouseFeatures):
    print(f"[REQ] Prediction request received: {feat.dict()}")
    if XGB_MODEL is None:
        print("[ERROR] Prediction failed: Model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_and_save.py first.")

    try:
        print("[INFO] Pre-processing...")
        df_row = _build_row(feat)
        X      = _preprocess(df_row)
        
        print("[INFO] Running inference...")
        log_pred = XGB_MODEL.predict(X)[0]
        price    = float(np.expm1(log_pred))
        
        # Clamp to a sane range
        price = max(50_000, min(price, 1_500_000))
        print(f"[OK] Prediction complete: ${price:,.2f}")
        return {"predicted_price": round(price, 2)}
    except Exception as exc:
        print(f"[ERROR] Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

# ── Mount static files at root (must be last)
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
