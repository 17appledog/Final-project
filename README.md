# 🏡 Ames House Price Predictor

AI-powered home valuation using **XGBoost** trained on the Ames Housing Dataset.  
- **91% R²** · **RMSE 0.115** (log scale) · **FastAPI** backend · **Glassmorphism** UI  
- No login required · Deploys to **Vercel** in minutes

---

## 📁 Project Structure

```
project/
├── api/
│   ├── __init__.py
│   └── predict.py          ← FastAPI serverless function
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/                 ← created by train_and_save.py
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   └── default_values.pkl
├── train.csv               ← Ames training data (you provide)
├── train_and_save.py       ← run this first!
├── requirements.txt
├── vercel.json
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ames-house-predictor.git
cd ames-house-predictor
pip install -r requirements.txt
```

### 2. Place your data

Copy `train.csv` (Ames Housing Dataset from Kaggle) into the project root.

### 3. Train and save the model

```bash
python train_and_save.py
```

This creates all `.pkl` files inside `models/`. Expect output like:
```
✅  All artefacts saved to ./models/
   Train RMSE (log): 0.0623   R²: 0.9712
```

### 4. Run the API locally

```bash
uvicorn api.predict:app --reload --port 8000
```

### 5. Open the frontend

Open `static/index.html` in your browser **or** serve it:

```bash
# Python quick server (serves from project root)
python -m http.server 3000
```

Then visit `http://localhost:3000/static/`.

> **Note:** For local dev, edit the first line of `script.js`:
> ```js
> const API_BASE = "http://localhost:8000";
> ```
> Change it back to `""` before deploying.

---

## ☁️ Deploy to Vercel

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Ames House Price Predictor"
git remote add origin https://github.com/YOUR_USERNAME/ames-house-predictor.git
git push -u origin main
```

### 2. Import on Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import your GitHub repository
3. Leave all settings as default — Vercel auto-detects `vercel.json`
4. Click **Deploy**

> **Important:** The `models/` folder must be committed to Git (the `.pkl` files).  
> Run `train_and_save.py` locally first, then commit the `models/` folder.

### 3. Verify

Visit your Vercel URL → the site loads → enter house details → click **Predict Price**.

---

## 🔌 API Reference

### `POST /api/predict`

**Request body (JSON):**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `OverallQual` | int | 1–10 | Overall material & finish quality |
| `GrLivArea` | float | 300–6000 | Above-grade living area (sq ft) |
| `Neighborhood` | string | see list | Ames city neighborhood |
| `YearBuilt` | int | 1872–2010 | Original construction year |
| `TotalSF` | float | 300–12000 | Total square footage |
| `HouseAge` | int | 0–150 | Age in years |
| `GarageCars` | int | 0–4 | Garage car capacity |
| `KitchenAbvGrd` | int | 0–3 | Kitchens above grade |
| `Fireplaces` | int | 0–3 | Number of fireplaces |
| `FullBath` | int | 0–4 | Full bathrooms above grade |
| `LotArea` | float | 1300–215000 | Lot area (sq ft) |
| `TotRmsAbvGrd` | int | 2–14 | Total rooms above grade |

**Response:**
```json
{ "predicted_price": 245000.00 }
```

### `GET /api/health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Regressor |
| Target | log1p(SalePrice) |
| RMSE (log scale) | ~0.115 |
| R² Score | ~0.91 |
| Training set | 1,460 homes |
| Hyperparameters | colsample_bytree=0.5, lr=0.01, max_depth=3, n_estimators=2000, subsample=0.7 |

### Preprocessing pipeline

1. Fill NaN: categorical → `"None"`, numeric → median
2. Feature engineering: TotalSF, TotalBath, HouseAge, OverallQual², etc.
3. Label-encode all categorical columns
4. Log1p-transform skewed features (|skewness| > 0.5)
5. RobustScaler (fit on training data only)

---

## 📝 Neighborhoods Reference

`Blmngtn` `Blueste` `BrDale` `BrkSide` `ClearCr` `CollgCr` `Crawfor` `Edwards`  
`Gilbert` `IDOTRR` `MeadowV` `Mitchel` `NAmes` `NoRidge` `NPkVill` `NridgHt`  
`NWAmes` `OldTown` `SWISU` `Sawyer` `SawyerW` `Somerst` `StoneBr` `Timber` `Veenker`

---

## 📄 License

MIT — free to use, modify, and deploy.
