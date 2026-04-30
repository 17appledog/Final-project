"""
train_and_save.py
-----------------
Reads train.csv, applies the exact same preprocessing as the notebook,
trains XGBoost, and saves:
  models/xgb_model.pkl
  models/scaler.pkl
  models/label_encoders.pkl
  models/feature_names.pkl
  models/default_values.pkl
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
DATA_PATH = "train.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Loading {DATA_PATH} …")
df = pd.read_csv(DATA_PATH)

# Drop Id column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Separate target
y = np.log1p(df["SalePrice"].values)
df = df.drop(columns=["SalePrice"])

# ──────────────────────────────────────────────
# 2. Fill missing values
# ──────────────────────────────────────────────
NONE_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
    "MasVnrType",
]
for col in NONE_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("None")

# Numeric → median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Remaining categoricals → mode
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ──────────────────────────────────────────────
# 3. Feature engineering (must mirror predict.py)
# ──────────────────────────────────────────────
df["TotalSF"]       = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBath"]     = (df["FullBath"] + 0.5 * df["HalfBath"] +
                       df.get("BsmtFullBath", 0) + 0.5 * df.get("BsmtHalfBath", 0))
df["TotalPorchSF"]  = (df["OpenPorchSF"] + df["EnclosedPorch"] +
                       df["3SsnPorch"] + df["ScreenPorch"])
df["HouseAge"]      = df["YrSold"] - df["YearBuilt"]
df["RemodelAge"]    = df["YrSold"] - df["YearRemodAdd"]
df["HasRemodel"]    = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
df["HasGarage"]     = (df["GarageArea"] > 0).astype(int)
df["HasBsmt"]       = (df["TotalBsmtSF"] > 0).astype(int)
df["HasFireplace"]  = (df["Fireplaces"] > 0).astype(int)
df["Has2ndFloor"]   = (df["2ndFlrSF"] > 0).astype(int)
df["HasWoodDeck"]   = (df["WoodDeckSF"] > 0).astype(int)
df["HasOpenPorch"]  = (df["OpenPorchSF"] > 0).astype(int)
df["HasPool"]       = (df["PoolArea"] > 0).astype(int)
df["LotPerSF"]      = df["LotArea"] / (df["TotalSF"] + 1)
df["GarageRatio"]   = df["GarageArea"] / (df["TotalSF"] + 1)
df["OverallQual2"]  = df["OverallQual"] ** 2
df["OverallQual3"]  = df["OverallQual"] ** 3
df["QualPerSF"]     = df["OverallQual"] / (df["TotalSF"] + 1)
df["YearBuilt2"]    = df["YearBuilt"] ** 2
df["TotalRooms"]    = df["TotRmsAbvGrd"] + df["FullBath"] + df["HalfBath"]
df["AvgRoomSize"]   = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)

# ──────────────────────────────────────────────
# 4. Log-transform skewed CONTINUOUS features
# ──────────────────────────────────────────────
# Identify truly continuous features (exclude ordinal, binary, and discrete counts)
cont_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
    # Engineered continuous features
    "TotalSF", "TotalPorchSF", "LotPerSF", "GarageRatio", "QualPerSF", "AvgRoomSize"
]
cont_features = [f for f in cont_features if f in df.columns]

skewness = df[cont_features].apply(lambda x: skew(x.dropna()))
high_skew = skewness[skewness.abs() > 0.5].index.tolist()
SKEW_COLS = high_skew  # save for predict.py reference

print(f"Applying log1p to {len(SKEW_COLS)} skewed continuous features...")
for col in SKEW_COLS:
    df[col] = np.log1p(df[col].clip(lower=0))

# ──────────────────────────────────────────────
# 5. Label-encode categoricals
# ──────────────────────────────────────────────
# Now categoricals are encoded AFTER log-transform checks, avoiding corruption
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders: dict[str, LabelEncoder] = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ──────────────────────────────────────────────
# 6. Scale
# ──────────────────────────────────────────────
feature_names = df.columns.tolist()
scaler = RobustScaler()
X = scaler.fit_transform(df)

# ──────────────────────────────────────────────
# 7. Compute default values (median / mode) for
#    features NOT exposed to the user
# ──────────────────────────────────────────────
default_values = {}
for col in df.columns:
    default_values[col] = float(df[col].median())

# ──────────────────────────────────────────────
# 8. Train XGBoost with Early Stopping
# ──────────────────────────────────────────────
# Split into train/validation for early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training XGBoost (Early Stopping, n_estimators=300) …")
xgb = XGBRegressor(
    colsample_bytree=0.5,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,   # Reduced; early stopping will handle convergence
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=1,           # reproducible results
    tree_method="hist", 
    early_stopping_rounds=20,
    verbosity=0,
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f"  Stopped at iteration: {xgb.best_iteration}")

# ──────────────────────────────────────────────
# 9. Save artefacts
# ──────────────────────────────────────────────
# Save in native XGBoost JSON format (faster load & predict than pickle)
xgb.save_model(os.path.join(MODELS_DIR, "xgb_model.json"))
joblib.dump(xgb,            os.path.join(MODELS_DIR, "xgb_model.pkl"))  # keep pkl as fallback
joblib.dump(scaler,         os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
joblib.dump(feature_names,  os.path.join(MODELS_DIR, "feature_names.pkl"))
joblib.dump({
    "default_values": default_values,
    "skew_cols": SKEW_COLS,
}, os.path.join(MODELS_DIR, "default_values.pkl"))

print("\n[OK] All artefacts saved to ./models/")
print(f"   Features : {len(feature_names)}")
print(f"   Skew cols: {len(SKEW_COLS)}")

# Quick sanity check
import sklearn.metrics as m
preds = xgb.predict(X)
rmse = np.sqrt(m.mean_squared_error(y, preds))
r2   = m.r2_score(y, preds)
print(f"   Train RMSE (log): {rmse:.4f}   R2: {r2:.4f}")
print("\nDone. Run the API with:  uvicorn api.predict:app --reload")
