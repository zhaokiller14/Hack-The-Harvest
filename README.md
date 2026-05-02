# Hack The Harvest

Smart agriculture API built for the Hack The Harvest hackathon. Detects olive grove stress, predicts tomato yield, and maps olive parcels using real Sentinel-2 satellite data.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Running the API](#running-the-api)
4. [Track 1 — Olive Grove Mapping](#track-1--olive-grove-mapping)
5. [Track 2 — Tomato Yield Prediction](#track-2--tomato-yield-prediction)
6. [Track 3 — Olive Anomaly Detection](#track-3--olive-anomaly-detection)
7. [Track 4 — Multimodal Darija Assistant](#track-4--multimodal-darija-assistant)
8. [API Contracts](#api-contracts)

---

## Project Structure

```text
app/
  main.py                            # FastAPI entry point, CORS, router registration
  api/
    cartographier.py                 # Track 1 — POST /api/cartographier
    rendement.py                     # Track 2 — POST /api/predire-rendement
    anomalie.py                      # Track 3 — POST /api/diagnostic-anomalie
  models/
    cartographier.py                 # Pydantic schemas (Track 1)
    rendement.py                     # Pydantic schemas (Track 2)
    anomalie.py                      # Pydantic schemas (Track 3)
  services/
    sentinel.py                      # GEE NDVI time series (cloud gaps interpolated)
    weather.py                       # Open-Meteo real weather fetcher
    ndvi.py                          # Numpy NDVI/NDWI/NDRE helpers
    parcel_baseline.py               # Prophet model loader & Z-score
    classifier.py                    # Olive grove classifier (Track 1)
    gee.py                           # GEE helpers

data/
  Oliviers/
    parcellesOliviersIntensifs.json  # 26 intensif parcels (Cap Bon, lat ~36.4)
    parcelles_OlivierExtensif.json   # 23 extensif parcels (Sfax, lat ~35.3)
  ndvi_history/
    <parcel_id>.csv                  # Sentinel-2 NDVI history per parcel (2019–2024)

models/
  prophet/
    <parcel_id>.json                 # One Prophet model per parcel (Track 3)
    meta.json                        # Training stats

scripts/
  build_ndvi_history.py              # Fetch GEE NDVI history for all parcels
  build_prophet_baselines.py         # Train Prophet models per parcel
  train_anomalie_baseline.py         # (legacy) Ridge fallback training
  train_classifier.py                # Train olive grove classifier (Track 1)
  extract_features.py                # Feature extraction from Sentinel-2
  prepare_data.py                    # Data preparation utilities

docs/
  track3-anomalie-README.md          # Detailed Track 3 documentation
```

---

## Environment Setup

> **Important:** the Python venv is at `venv/` — always use `venv/bin/python` and `venv/bin/uvicorn`, not system `python`.

```bash
# Install dependencies
venv/bin/pip install -r requirements.txt

# Authenticate Google Earth Engine (opens browser once per machine)
venv/bin/python -c "import ee; ee.Authenticate()"

# Set GEE project in .env
echo "GEE_PROJECT=your-gcp-project-id" >> .env

# Verify GEE works
venv/bin/python -c "import ee; ee.Initialize(project='your-gcp-project-id'); print(ee.String('ok').getInfo())"
```

---

## Running the API

```bash
venv/bin/uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Track 1 — Olive Grove Mapping

**Endpoint:** `POST /api/cartographier`

Classifies olive grove parcels from Sentinel-2 imagery using a U-Net segmentation + RF/XGBoost pipeline.

**Train the classifier:**

```bash
venv/bin/python scripts/extract_features.py
venv/bin/python scripts/train_classifier.py
```

---

## Track 2 — Tomato Yield Prediction

**Endpoint:** `POST /api/predire-rendement`

Predicts tomato yield using LightGBM/XGBoost with SHAP feature explanations and conformal prediction intervals.

---

## Track 3 — Olive Anomaly Detection

**Endpoint:** `POST /api/diagnostic-anomalie`

Detects early stress (2–3 weeks ahead) on olive grove parcels by comparing the **observed NDVI** from Sentinel-2 against a **Prophet per-parcel baseline** built from the parcel's own 6-year satellite history.

The approach is fully **unsupervised** — the input data has no labels, no NDVI values, only polygon geometries. The baseline is learned entirely from each parcel's real satellite history.

### Build the baselines (run once, ~25 min)

```bash
# Step 1 — fetch Sentinel-2 NDVI history for all 49 parcels via GEE (~20 min)
venv/bin/python scripts/build_ndvi_history.py

# Step 2 — train one Prophet model per parcel (~5 min)
venv/bin/python scripts/build_prophet_baselines.py
```

### How it works

1. **Observed NDVI** — GEE queries `COPERNICUS/S2_SR_HARMONIZED` for a 5-step bi-weekly window ending at the requested date. Cloud gaps are filled by linear interpolation.

2. **Expected NDVI** — Prophet forecasts what the parcel should show, accounting for:
   - Its annual seasonal cycle (spring green-up, summer stress dip)
   - Its long-term trend (slow greening or drying over years)

   A simple monthly mean assumes NDVI is stationary across years — wrong for maturing olive groves. Prophet separates trend from seasonality so July 2024 expectations differ from July 2019.

3. **Anomaly score** — Z-score averaged over the 5-step window:
   ```
   ```text
   sigma = (yhat_upper - yhat_lower) / 3.92   # Prophet 95% CI → std
   Z     = (expected - observed) / sigma       per step
   score = mean(max(Z, 0) across 5 steps)
   ```

   `score < 1.0` → vert, `< 2.0` → orange, `≥ 2.0` → rouge

See [docs/track3-anomalie-README.md](docs/track3-anomalie-README.md) for full details.

### Quick test

```bash
curl -s -X POST http://localhost:8000/api/diagnostic-anomalie \
  -H "Content-Type: application/json" \
  -d '{
    "oliveraie": {
      "id": "parcel_1777662049508_exj14",
      "polygone": {
        "type": "Polygon",
        "coordinates": [[[10.015, 36.448], [10.008, 36.450], [10.006, 36.442], [10.015, 36.448]]]
      },
      "systeme": "intensif"
    },
    "date": "2024-07-15"
  }' | python3 -m json.tool
```

---

## Track 4 — Multimodal Darija Assistant

A separate application (not part of this FastAPI). Uses Whisper + CNN + RAG + TTS to answer farmer queries in Tunisian Arabic (Darija).

---

## API Contracts

### Track 1 — `POST /api/cartographier`

Response items are **flat** — fields `polygone`, `systeme`, `confiance`, `surface_ha` are at the top level, not nested under `properties`.

### Track 2 — `POST /api/predire-rendement`

- `intervalle_confiance_95` is a **2-element `list[float]`**, not a dict
- SHAP feature field is `impact` (signed float), not `importance`/`direction`

### Track 3 — `POST /api/diagnostic-anomalie`

- `ndvi_observe` and `ndvi_attendu` are **`list[float]`** time series (5 values, oldest → newest)
- `anomaly_score` is **unbounded** (e.g. 2.4), not clamped to 0–1
