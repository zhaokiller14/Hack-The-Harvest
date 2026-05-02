# Track 3 — Stress hydrique & santé des cultures

## Détection précoce d'anomalies sur oliveraies

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Input Data](#input-data)
4. [How Detection Works](#how-detection-works)
5. [Unsupervised Path — Prophet Per-parcel Model](#unsupervised-path--prophet-per-parcel-model)
6. [Weather Service](#weather-service)
7. [NDVI Service](#ndvi-service)
8. [Anomaly Scoring](#anomaly-scoring)
9. [API Contract](#api-contract)
10. [Setup & Running Order](#setup--running-order)

---

## Overview

The system detects anomalies in olive grove health **2–3 weeks early** by comparing the **observed NDVI** of a parcel (from real Sentinel-2 satellite data) against the **expected NDVI** (what a healthy parcel should show given the season, weather, and multi-year trend).

```text
anomaly = observed NDVI  vs  expected NDVI
```

The data in `data/Oliviers/` contains **no labels** — no "stressed" / "healthy" tags, no NDVI values, only polygon geometries. The problem is therefore **unsupervised**: the baseline is built entirely from the parcel's own 6-year satellite history.

---

## Repository Structure

```text
app/
  api/anomalie.py                    # POST /api/diagnostic-anomalie endpoint
  models/anomalie.py                 # Pydantic request/response schemas
  services/
    weather.py                       # Open-Meteo real weather fetcher
    sentinel.py                      # GEE NDVI time series (cloud gaps interpolated)
    ndvi.py                          # Numpy NDVI/NDWI/NDRE helpers
    parcel_baseline.py               # Prophet model loader & Z-score

data/
  Oliviers/
    parcellesOliviersIntensifs.json  # 26 intensif parcels (Cap Bon, lat ~36.4)
    parcelles_OlivierExtensif.json   # 23 extensif parcels (Sfax, lat ~35.3)
  ndvi_history/
    <parcel_id>.csv                  # fetched by build_ndvi_history.py (GEE)

models/
  prophet/
    <parcel_id>.json                 # one Prophet model per parcel
    meta.json                        # parcel list + training stats

scripts/
  build_ndvi_history.py              # Step 1 — fetch GEE history for all parcels
  build_prophet_baselines.py         # Step 2 — train one Prophet model per parcel
```

---

## Input Data

### What the parcel JSON files contain

Two files in `data/Oliviers/`:

| File | System | Count | Region |
| --- | --- | --- | --- |
| `parcellesOliviersIntensifs.json` | `intensif` | 26 | Nabeul / Cap Bon |
| `parcelles_OlivierExtensif.json` | `extensif` | 23 | Sfax / Kairouan |

Each parcel object:

```json
{
  "id": "parcel_1777662049508_exj14",
  "area_ha": 74.01,
  "coordinates": [
    { "lat": 36.448, "lng": 10.015 },
    { "lat": 36.450, "lng": 10.008 },
    ...
  ]
}
```

**These files contain no NDVI values, no labels, no measurements** — only polygon geometries and area. They are used exclusively to:

- Get the polygon for GEE spatial queries
- Get the centroid for weather API queries

The `systeme` field is inferred from the filename (`intensif` / `extensif`), or provided directly in the API request.

### Converting coordinates to GeoJSON

The parcel JSON uses `{lat, lng}`. GEE and the API expect GeoJSON `[longitude, latitude]`:

```python
ring = [[c["lng"], c["lat"]] for c in parcel["coordinates"]]
ring.append(ring[0])  # close the polygon ring
geojson = {"type": "Polygon", "coordinates": [ring]}
```

---

## How Detection Works

The endpoint answers one question:

> *Is this parcel's NDVI lower than Prophet expects at this point in its 6-year trajectory?*

```text
ndvi_observe  = what the parcel has right now       (from GEE satellite)
ndvi_attendu  = what Prophet predicts it should be  (from its own history + trend)

anomaly_score = (expected − observed) / Prophet_uncertainty   (Z-score)
```

If the score exceeds a threshold → alert.

---

## Unsupervised Path — Prophet Per-parcel Model

This is the correct approach for the problem. No labels, no training data — only the parcel's own satellite observations.

### Why Prophet, not a simple monthly mean?

A naive approach would be: *group all July observations, compute their mean, use that as the July baseline*.

The problem is that this assumes the parcel NDVI is **stationary across years** — i.e., July 2019 and July 2024 should show the same NDVI. This is wrong:

- Olive groves slowly change as trees mature, soil evolves, or irrigation improves
- A parcel may be slowly greening (+trend) or slowly drying (−trend) over 6 years
- Using a flat July mean would flag a naturally greening parcel as anomalous in recent years

**Prophet solves this** by modelling two things simultaneously from the historical time series:

1. **Seasonal pattern** — how NDVI rises and falls through the year (spring green-up, summer stress dip, etc.)
2. **Long-term trend** — whether the parcel is slowly greening or drying over the years

This means the expected NDVI for July 2024 is **not the same** as July 2019 — Prophet adjusts for the multi-year drift. A simple monthly mean cannot do this.

### Step 1 — Fetch NDVI history from GEE

`scripts/build_ndvi_history.py` queries all 49 parcels against Sentinel-2 (2019–2024):

```text
For each parcel:
  → GEE: filter Sentinel-2 scenes over the polygon, cloud < 30%
  → compute mean NDVI per scene over the polygon interior at 10m resolution
  → save: data/ndvi_history/<parcel_id>.csv
```

Each CSV contains one row per cloud-free satellite pass:

```csv
date,       doy,  ndvi_mean
2019-03-15, 74,   0.4821
2019-04-02, 92,   0.5103
2019-04-16, 106,  0.5340
...
```

A typical parcel accumulates **80–150 valid observations** over 6 years (Sentinel-2 revisit is 5 days, but cloud cover reduces this).

Run it:

```bash
venv/bin/python scripts/build_ndvi_history.py           # all parcels (~20 min)
venv/bin/python scripts/build_ndvi_history.py --resume  # skip already-done parcels
```

### Step 2 — Train one Prophet model per parcel

`scripts/build_prophet_baselines.py` trains a Prophet model on each parcel's historical NDVI:

```python
model = Prophet(
    yearly_seasonality=True,        # learns the annual NDVI cycle
    weekly_seasonality=False,       # no weekly pattern in olive NDVI
    daily_seasonality=False,
    seasonality_mode="multiplicative",  # seasonal swings scale with the trend level
    changepoint_prior_scale=0.05,       # slow smooth trend (not abrupt jumps)
    seasonality_prior_scale=10.0,       # allow flexible seasonal shape
    interval_width=0.95,                # 95% uncertainty interval
)
model.fit(df)  # df has columns: ds (datetime), y (NDVI)
```

Prophet requires at least 24 observations (≥2 years) to reliably separate trend from seasonality.

Output: `models/prophet/<parcel_id>.json` — one JSON file per parcel.

Run it:

```bash
venv/bin/python scripts/build_prophet_baselines.py
```

### Step 3 — Predict expected NDVI at inference

At inference, `app/services/parcel_baseline.py` loads the Prophet model and predicts the expected NDVI for each of the 5 observation dates:

```python
future = pd.DataFrame({"ds": pd.to_datetime(dates)})
forecast = model.predict(future)

# Extract mean and uncertainty for each date
mean = forecast["yhat"]
std  = (forecast["yhat_upper"] - forecast["yhat_lower"]) / 3.92
# 95% CI → sigma: CI_width / (2 × 1.96)
```

The `std` (sigma) comes from Prophet's 95% uncertainty interval, which grows wider for dates far from the training period or in seasons with high historical variability.

Example:

- Prophet expects NDVI = **0.50 ± 0.03** in mid-July for this parcel
- Satellite today measures **0.37**
- Z-score = (0.50 − 0.37) / 0.03 = **4.3 → rouge**

Only the parcel's own history can tell you that — a generic model trained on Tunisia phenology has no way of knowing this specific parcel sits at 0.50 in July.

---

## Weather Service

`app/services/weather.py` fetches real weather from **Open-Meteo** (no API key required).

For each of the 5 bi-weekly time steps, it fetches a 21-day window and returns:

| Variable | Description |
| --- | --- |
| `rainfall_mm` | Cumulative precipitation (mm) |
| `et0_mm` | Cumulative evapotranspiration (mm) |
| `gdd` | Growing degree days (base 10°C) |
| `heat_stress_days` | Days with Tmax > 35°C |

**Date routing:**

- Past dates → Open-Meteo Archive API
- Up to +16 days → Open-Meteo Forecast API
- Further future → same period shifted −1 year (climate proxy)

The −1 year shift is needed for requests like `"date": "2026-07-15"` which is in the future relative to today.

---

## NDVI Service

`app/services/sentinel.py` returns the 5-step observed NDVI series via **Google Earth Engine only**.

For each of the 5 bi-weekly steps:

- Queries `COPERNICUS/S2_SR_HARMONIZED` for a ±7-day window around the step date
- Filters scenes with cloud cover < 30%
- Computes `(B8 − B4) / (B8 + B4)` per pixel, takes the mean over the polygon at 10 m resolution
- Returns one float per step — **no raster is downloaded**, all computation runs on Google's servers

**Cloud gaps** (a step where every scene in the window is too cloudy) are filled automatically by **linear interpolation** between the neighbouring real satellite values. If all 5 steps are cloudy the endpoint raises an error.

**Future dates** (e.g. `"2026-07-15"`) are handled by shifting back 1 year before querying GEE, since the archive only covers past dates.

---

## Anomaly Scoring

```python
sigma = (yhat_upper - yhat_lower) / 3.92    # Prophet 95% CI → std
Z     = (expected − observed) / sigma        per step
score = mean(max(Z, 0) across 5 steps)
```

The score is measured in units of the parcel's own uncertainty — a Z-score of 2 means the NDVI is 2 standard deviations below what Prophet expected.

Threshold: score < 1.0 → vert, < 2.0 → orange, ≥ 2.0 → rouge

---

## API Contract

### Request

```json
POST /api/diagnostic-anomalie
{
  "oliveraie": {
    "id": "parcel_1777662049508_exj14",
    "polygone": {
      "type": "Polygon",
      "coordinates": [[[10.015, 36.448], [10.008, 36.450], [10.006, 36.442], [10.015, 36.448]]]
    },
    "systeme": "intensif"
  },
  "date": "2026-07-15"
}
```

`polygone` — GeoJSON Polygon, `[longitude, latitude]` order.
`systeme` — `"extensif"` | `"intensif"` | `"hyper_intensif"`.
`date` — any ISO date, past or future.

### Response

```json
{
  "statut": "orange",
  "anomaly_score": 1.84,
  "ndvi_observe": [0.535, 0.540, 0.429, 0.441, 0.370],
  "ndvi_attendu": [0.562, 0.560, 0.488, 0.458, 0.418],
  "explication": "NDVI 11% en dessous attendu malgré déficit hydrique marqué. Stress modéré détecté sur intensif.",
  "recommandation": "Inspection visuelle dans 48h. Vérifier le système d'irrigation si applicable."
}
```

`ndvi_observe` and `ndvi_attendu` — 5 values ordered **oldest → newest** (bi-weekly, ~10 weeks back).

---

## Setup & Running Order

### First time setup

```bash
# 1. Install dependencies
venv/bin/pip install -r requirements.txt

# 2. Authenticate GEE (opens browser once)
venv/bin/python -c "import ee; ee.Authenticate()"

# 3. Set GEE project in .env
echo "GEE_PROJECT=your-gcp-project-id" >> .env

# 4. Verify GEE works
venv/bin/python -c "import ee; ee.Initialize(project='your-gcp-project-id'); print(ee.String('ok').getInfo())"
```

### Build the unsupervised baselines (run once, ~25 min)

```bash
# Step 1 — Fetch real Sentinel-2 NDVI history for all 49 parcels (~20 min)
venv/bin/python scripts/build_ndvi_history.py

# Step 2 — Train one Prophet model per parcel (~5 min)
venv/bin/python scripts/build_prophet_baselines.py
```

After this, the endpoint is ready for all 49 parcels.

### Start the API

```bash
venv/bin/uvicorn app.main:app --reload
```
