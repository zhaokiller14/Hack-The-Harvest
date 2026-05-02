# Track 2 — Prédiction de rendement tomate

## Yield Prediction for Tunisian Tomato Parcels

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Input Data](#input-data)
4. [Model Architecture](#model-architecture)
5. [Feature Engineering](#feature-engineering)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Data Sources at Inference](#data-sources-at-inference)
9. [API Contract](#api-contract)
10. [Setup & Running Order](#setup--running-order)

---

## Overview

Given a tomato parcel polygon and a prediction date, the system returns:

- **Predicted yield** in t/ha
- **95% confidence interval** (MC Dropout uncertainty quantification)
- **Top 5 influential features** (gradient-based sensitivity, SHAP-style)
- **Estimated harvest date** (plantation + ~120 days)
- **Uncertainty flag** (NORMALE / HAUTE)

The model is a **MLP with MC Dropout** trained on 243 labeled Tunisian tomato parcels. At inference it runs 50 stochastic forward passes with dropout active, producing a distribution over predictions whose mean and std define the confidence interval.

---

## Repository Structure

```text
app/
  api/rendement.py            # POST /api/predire-rendement endpoint
  models/rendement.py         # Pydantic request/response schemas
  services/
    mlp_model.py              # MCDropoutMLP architecture + RendementPredictor
    feature_builder.py        # Raw API outputs → 187 z-scored features
    weather.py                # Open-Meteo 4-season weather fetcher
    sentinel.py               # Sentinel-2 vegetation indices
    soil.py                   # SoilGrids soil properties fetcher

scripts/
  train_mlp.py                # Train MLP on TomateProduction data → mlp_model.pt

mlp_model.pt                  # Trained model checkpoint (weights + scaler + feature names)

TomateProduction/
  dev_set_07.csv              # Training set (~195 parcels)
  hold_set_07.csv             # Hold-out set (~48 parcels)
```

---

## Input Data

### TomateProduction dataset

Two CSVs with pre-computed features and labeled yield:

| File | Role | Parcels |
| --- | --- | --- |
| `dev_set_07.csv` | Training + CV | ~195 |
| `hold_set_07.csv` | Hold-out evaluation | ~48 |

**Target column:** `rendement_tha` — yield in tonnes per hectare.

**Feature columns (~201 raw → ~187 after cleaning):**

The dataset already contains the engineered features (weather per season, vegetation indices, soil, interactions). The training script drops 14 near-constant features (>80% same value) before fitting.

Key feature groups:

| Group | Count | Description |
| --- | --- | --- |
| Weather × 4 seasons | 72 | GDD, temperature, precipitation, ET0, humidity, heat/drought stress |
| Vegetation indices × 4 seasons | 56 | NDVI, EVI, EVI2, DSWI, NDWI, NRI (mean, max, std) |
| Soil static | 12 | Clay, sand, silt, pH, SOC, CEC, AWC, texture index |
| Delta features | 9 | Season-to-season change in NDVI, humidity, water stress |
| Fuzzy binary flags | 16 | Drought, low NDVI, heat, cold per season |
| Score features | 5 | Humidity deficit, water stress, heat, combined stress scores |
| Interaction features | 16 | NDVI×EVI ratio, NDVI×GDD, drought×NDVI, heat×drought |
| Static | 1 | Parcel area (ha) |

---

## Model Architecture

### MCDropoutMLP

```text
Input (187 features)
  → Linear(187, 256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256, 128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128, 64)  → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64, 1)
  → output: log(1 + yield_t/ha)
```

### MC Dropout uncertainty

At inference, dropout layers stay **active** (`.train()` mode). Running 50 forward passes through the same input gives 50 different predictions due to random neuron dropout:

```python
preds = [model(x) for _ in range(50)]   # 50 stochastic passes
mean_log = mean(preds)                   # point estimate in log space
std_log  = std(preds)                    # uncertainty in log space

tonnage   = exp(mean_log)
ci_low    = exp(mean_log - 1.96 × std_log)   # 95% CI lower
ci_high   = exp(mean_log + 1.96 × std_log)   # 95% CI upper
```

Uncertainty flag: `std_log > 0.45` or prediction outside [30, 120] t/ha → `"HAUTE"`.

---

## Feature Engineering

`app/services/feature_builder.py` converts raw API outputs into the 187 features the model expects.

The dataset was built from physical measurements. At inference, we reconstruct the same features from live APIs:

```text
fetch_weather()   → weather dict  { "s1": {...}, "s2": {...}, "s3": {...}, "s4": {...} }
fetch_sentinel2() → bands dict    { "ndvi_series": [...], "evi_series": [...], ... }
fetch_soil()      → soil dict     { "clay": 28.0, "sand": 42.0, "phh2o": 7.6, ... }
```

Each raw value is converted to a **z-score** relative to reference distributions calibrated on typical Tunisian tomato conditions. This puts all features in the same scale as the training data.

### The 4 phenological seasons

Weather and vegetation features are split into 4 × 30-day windows starting from `date_plantation`:

| Season | Days since planting | Stage |
| --- | --- | --- |
| s1 | 0–30 | Establishment |
| s2 | 30–60 | Vegetative growth |
| s3 | 60–90 | Flowering / fruit set |
| s4 | 90–120 | Fruit fill / maturation |

---

## Training Pipeline

`scripts/train_mlp.py`:

```text
1. Load dev_set_07.csv + hold_set_07.csv → 243 parcels
2. Drop 14 near-constant features → 187 features
3. Log-transform target: y = log1p(rendement_tha)
4. Fit StandardScaler on full data (saved in checkpoint)
5. Stratified 5-fold CV (low/med/high yield strata)
   - Oversample low-yield class 3× in each training fold
   - Train MCDropoutMLP: Huber loss, Adam lr=1e-3, cosine LR, early stopping (patience=20)
   - Report per-fold MAPE
6. Print OOF MAPE by yield class
7. Retrain final model on full data (300 epochs, same oversampling)
8. Save checkpoint: mlp_model.pt
   { model_state, scaler_mean, scaler_std, feat_cols }
```

Run training:

```bash
venv/bin/python scripts/train_mlp.py
```

The data path in the script points to `TomateProduction/`. Adjust if needed.

---

## Inference Pipeline

```text
POST /api/predire-rendement
        ↓
rendement.py endpoint
        ↓
┌──────────────────────────────────────────┐
│  fetch_weather()   → Open-Meteo archive  │  ← 4 seasons × 17 weather vars
│  fetch_sentinel2() → Sentinel-2 / GEE   │  ← vegetation indices
│  fetch_soil()      → SoilGrids REST API  │  ← 9 soil properties
└──────────────────────────────────────────┘
        ↓
build_features()  →  187 z-scored features
        ↓
RendementPredictor.predict()
  → normalize → MC Dropout (50 passes) → mean ± std in log space
  → exp() → tonnage + 95% CI
  → gradient sensitivity → top 5 SHAP-style features
        ↓
RendementResponse
```

---

## Data Sources at Inference

### Weather — Open-Meteo (no API key)

`app/services/weather.py` fetches the full growing season from the archive API and aggregates into 4 × 30-day windows:

- GDD (base 10°C), temperature stats, precipitation, ET0, solar radiation, humidity, heat/drought stress
- Falls back to Tunisian climatological defaults on error

### Vegetation indices — Sentinel-2

`app/services/sentinel.py` returns NDVI and derived indices per season.

> **Note:** on this branch, `sentinel.py` is a stub returning empty band arrays. The feature builder gracefully falls back to dataset mean values when bands are empty, so the endpoint still works — it just predicts from weather + soil only.

### Soil — SoilGrids REST API (no API key)

`app/services/soil.py` fetches 9 topsoil properties (0–30 cm average) for the parcel centroid:

- Clay, sand, silt, pH, SOC, CEC, bulk density, nitrogen, coarse fragments
- Derives AWC (available water capacity) via Saxton–Rawls approximation
- Falls back to Tunisian median values on error

---

## API Contract

### Request

```json
POST /api/predire-rendement
{
  "parcelle": {
    "id": "parcel_xyz",
    "polygone": {
      "type": "Polygon",
      "coordinates": [[[10.015, 36.448], [10.008, 36.450], [10.006, 36.442], [10.015, 36.448]]]
    },
    "date_plantation": "2024-03-01",
    "variete": "Rio Grande"
  },
  "date_prediction": "2024-06-01"
}
```

### Response

```json
{
  "tonnage_predit_t": 74.3,
  "intervalle_confiance_95": [58.1, 94.7],
  "top_features_shap": [
    { "feature": "stress_hydrique_s3", "impact": -0.312 },
    { "feature": "ndvi_mean_s2",       "impact":  0.245 },
    { "feature": "gdd_s4",             "impact":  0.198 },
    { "feature": "clay",               "impact": -0.134 },
    { "feature": "intensite_chaleur_s3","impact": -0.121 }
  ],
  "date_recolte_estimee": "2024-06-29",
  "incertitude": {
    "niveau": "NORMALE",
    "sigma_log": 0.21
  }
}
```

**Key contract rules (must match exactly):**
- `intervalle_confiance_95` — 2-element `list[float]` `[min, max]`, not a dict
- `top_features_shap[].impact` — signed float (positive = boosts yield, negative = reduces it)
- `tonnage_predit_t` — yield in **t/ha** (not total tonnes)

---

## Setup & Running Order

### Install dependencies

```bash
venv/bin/pip install -r requirements.txt
```

### Train the model (requires TomateProduction data)

```bash
venv/bin/python scripts/train_mlp.py
# Output: mlp_model.pt
```

### Start the API

```bash
venv/bin/uvicorn app.main:app --reload
```

### Quick test

```bash
curl -s -X POST http://localhost:8000/api/predire-rendement \
  -H "Content-Type: application/json" \
  -d '{
    "parcelle": {
      "id": "test_parcel",
      "polygone": {
        "type": "Polygon",
        "coordinates": [[[10.015, 36.448], [10.008, 36.450], [10.006, 36.442], [10.015, 36.448]]]
      },
      "date_plantation": "2024-03-01",
      "variete": "Rio Grande"
    },
    "date_prediction": "2024-06-01"
  }' | python3 -m json.tool
```
