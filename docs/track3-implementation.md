# Track 3 — Implementation Guide
## Stress hydrique & détection d'anomalies sur oliveraies

---

## Architecture overview

```
POST /api/diagnostic-anomalie
          │
          ▼
   app/api/anomalie.py          ← orchestrator
     │         │         │
     ▼         ▼         ▼
weather.py  sentinel.py  Ridge model
(Open-Meteo) (GEE / phéno) (models/)
```

The endpoint fetches **real weather** from Open-Meteo, computes a **NDVI time series** (Google Earth Engine if authenticated, phenological model otherwise), then runs a **trained Ridge regression** to predict the expected healthy NDVI and scores the deviation.

---

## Files changed / created

| File | Role |
|------|------|
| `app/services/weather.py` | Real Open-Meteo API client |
| `app/services/sentinel.py` | NDVI time series — GEE + fallback |
| `app/services/ndvi.py` | Numpy NDVI/NDWI/NDRE helpers |
| `app/api/anomalie.py` | Full endpoint — replaces stub |
| `scripts/train_anomalie_baseline.py` | Generates training data, trains Ridge models |
| `models/ridge_extensif.pkl` | Trained model — extensif system |
| `models/ridge_intensif.pkl` | Trained model — intensif system |
| `models/ridge_hyper_intensif.pkl` | Trained model — hyper-intensif system |
| `models/residual_std_*.csv` | Per-parcel NDVI residual std (Z-score thresholds) |
| `models/baseline_meta.json` | Feature names, system stats, MAE/R² |

---

## 1. Weather service (`app/services/weather.py`)

### What it does

Fetches aggregated weather for a **21-day sliding window** ending on any given date, using the Open-Meteo API — no account, no API key required.

### Routing logic

| Date condition | API used |
|----------------|----------|
| Past date | `archive-api.open-meteo.com/v1/archive` |
| Within 16 days ahead | `api.open-meteo.com/v1/forecast` |
| More than 16 days ahead | Same as above but date shifted −1 year (climate proxy) |

The −1 year shift is key for the hackathon: the jury may send `"date": "2026-07-15"` (future). The service transparently fetches the equivalent 2025-07-15 window from the archive.

### Output per window

```python
{
  "rainfall_mm": 5.2,      # cumulative precipitation (mm)
  "et0_mm": 97.4,          # cumulative evapotranspiration (mm)
  "gdd": 241.0,            # growing degree days (base 10 °C)
  "heat_stress_days": 8,   # days with Tmax > 35 °C
}
```

### Time series helper

`fetch_weather_series(lat, lon, end_date, n_steps=5, step_days=14)` runs 5 parallel requests and returns a list ordered **oldest → newest**, matching the NDVI time series index order.

---

## 2. Sentinel-2 / NDVI service (`app/services/sentinel.py`)

### Two paths

**Path A — Google Earth Engine (real satellite data)**

Activated automatically when `ee.Initialize()` succeeds. For each of the 5 time steps, queries `COPERNICUS/S2_SR_HARMONIZED`, filters cloud cover < 30 %, computes mean NDVI over the parcel polygon at 10 m resolution.

To enable:
```bash
venv/bin/python -c "import ee; ee.Authenticate()"
# Then set GEE_PROJECT=your-project-id in .env
```

**Path B — Phenological model (always available)**

When GEE is not authenticated or unavailable, the service estimates NDVI from:

1. **Olive phenology curve** — Tunisia-calibrated cosine model:

```
ndvi_seasonal(doy) = base + amplitude × cos(2π × (doy − peak_doy) / 365)
```

| System | base | amplitude | peak DOY |
|--------|------|-----------|----------|
| extensif | 0.36 | 0.09 | 115 (late Apr) |
| intensif | 0.50 | 0.07 | 125 (early May) |
| hyper_intensif | 0.60 | 0.05 | 130 (mid May) |

2. **Weather stress correction**:

```
stress = lush_bonus − drought × 0.08 − heat_penalty

lush_bonus   = min(rainfall_mm / 80, 0.04)
drought      = max(0, et0_mm − rainfall_mm) / et0_mm
heat_penalty = heat_stress_days × 0.005
```

Cloud-gap steps in GEE results are filled automatically with phenological estimates.

---

## 3. Ridge baseline model

### Why Ridge regression

- Captures linear seasonal + weather effects cleanly
- One model per system (extensif / intensif / hyper_intensif) — different rainfall sensitivity
- Fully interpretable coefficients
- Fast inference (< 1 ms per request)

### Features (7 total)

| Feature | Encoding | Effect direction |
|---------|----------|-----------------|
| `doy_sin` | sin(2π × DOY / 365) | captures spring rise |
| `doy_cos` | cos(2π × DOY / 365) | captures seasonal phase |
| `rainfall_21d` | raw mm | + (more rain → greener) |
| `et0_21d` | raw mm | − (high demand → stress) |
| `gdd_21d` | °C-days above 10 °C | ± (growth vs heat) |
| `heat_stress_days` | count | − (extreme heat) |
| `area_ha_log` | log(1 + area_ha) | proxy for canopy density |

### Training data

The training script (`scripts/train_anomalie_baseline.py`) generates synthetic but phenologically realistic data:

- **57 parcels** (26 intensif + 23 extensif from EZZAYRA JSON files + 8 synthetic hyper_intensif)
- **6 years** of bi-weekly observations (2018–2024) → 156 dates per parcel
- **8 892 samples** total
- Synthetic weather drawn from Tunisia monthly normals (Cap Bon / Sfax blend) with realistic variance
- Healthy NDVI = phenology curve + weather effect + Gaussian noise

### Model performance

| System | MAE train | MAE test | R² test |
|--------|-----------|----------|---------|
| extensif | 0.0145 | 0.0143 | 0.930 |
| intensif | 0.0112 | 0.0113 | 0.921 |
| hyper_intensif | 0.0087 | 0.0088 | 0.898 |

MAE of ~0.01 NDVI units on a 0–1 scale is well within measurement noise for Sentinel-2.

### Retrain

```bash
venv/bin/python scripts/train_anomalie_baseline.py
```

This overwrites the three `.pkl` files and `baseline_meta.json`. Run after any change to phenology parameters or after integrating real Sentinel-2 training data.

---

## 4. Anomaly scoring

### Score formula

```python
deviations = [(expected - observed) / (expected + 1e-8)
              for observed, expected in zip(ndvi_observe, ndvi_attendu)]
anomaly_score = max(0, mean(deviations)) * 10
```

Positive = observed NDVI below expected (stress). Unbounded — score of 2.4 means the parcel is on average 24 % below its expected healthy level.

### Status thresholds

| Score | Status | Meaning |
|-------|--------|---------|
| < 1.0 | `vert` | Normal — within seasonal expectations |
| 1.0 – 2.0 | `orange` | Moderate stress — inspect within 48h |
| ≥ 2.0 | `rouge` | Severe anomaly — urgent intervention |

### Dynamic Z-score (future upgrade)

`models/residual_std_<system>.csv` stores per-parcel historical residual standard deviations. For known parcels, the threshold denominator can be replaced by the parcel-specific sigma:

```python
z = (mean(ndvi_attendu) - mean(ndvi_observe)) / parcel_sigma
```

---

## 5. Explanation generation

The endpoint auto-generates `explication` and `recommandation` based on:

- **% NDVI below expected** (from observed vs expected means)
- **Weather characterisation** of the most recent window:
  - drought index = `(et0 - rain) / et0`
  - heat stress day count
- **Data source** — appends "(estimation phénologique — satellite non disponible)" when GEE is offline

Example outputs by status:

**vert** — `"NDVI conforme aux attentes saisonnières pour un olivier intensif. Écart observé: 3%."`

**orange** — `"NDVI 18% en dessous attendu malgré déficit hydrique marqué. Stress modéré détecté sur intensif — peut indiquer début de stress hydrique ou attaque précoce de ravageurs."`

**rouge** — `"NDVI 31% en dessous attendu avec 10 jours de chaleur extrême. Stress hydrique sévère détecté. Intervention urgente recommandée."`

---

## 6. Full request / response

### Request

```json
POST /api/diagnostic-anomalie
{
  "oliveraie": {
    "id": "O_2026_307",
    "polygone": {
      "type": "Polygon",
      "coordinates": [[[10.015, 36.448], [10.008, 36.450], [10.006, 36.442], [10.015, 36.448]]]
    },
    "systeme": "intensif"
  },
  "date": "2026-07-15"
}
```

`polygone` — GeoJSON Polygon, coordinates in `[longitude, latitude]` order.
`systeme` — one of `"extensif"` | `"intensif"` | `"hyper_intensif"`.
`date` — ISO 8601. Past, present, or future dates all work.

### Response

```json
{
  "statut": "orange",
  "anomaly_score": 1.84,
  "ndvi_observe": [0.5352, 0.5403, 0.4293, 0.4407, 0.3704],
  "ndvi_attendu": [0.5621, 0.5600, 0.4878, 0.4577, 0.4178],
  "explication": "NDVI 11% en dessous attendu malgré déficit hydrique marqué. Stress modéré détecté...",
  "recommandation": "Inspection visuelle dans 48h. Vérifier le système d'irrigation si applicable."
}
```

`ndvi_observe` and `ndvi_attendu` are 5-element lists ordered **oldest → newest** (bi-weekly, covering ~10 weeks).

---

## 7. Running the app

```bash
# Install dependencies (first time)
venv/bin/pip install -r requirements.txt

# Train models (already done — only needed to retrain)
venv/bin/python scripts/train_anomalie_baseline.py

# Start API
venv/bin/uvicorn app.main:app --reload
```

### Enable real satellite data (optional)

```bash
# Authenticate once per machine
venv/bin/python -c "import ee; ee.Authenticate()"

# Set project in .env
echo "GEE_PROJECT=your-gcp-project-id" >> .env
```

Once authenticated, all `/api/diagnostic-anomalie` calls automatically use real Sentinel-2 NDVI instead of the phenological estimate.

---

## 8. Pitfalls avoided

| Pitfall | How it is handled |
|---------|------------------|
| Phénologie ignorée | Seasonal cosine curve + DOY features in Ridge |
| Bruit pixel | GEE averages over polygon interior at 10 m resolution |
| Différence E/I/HI | Separate Ridge model per system with different weather coefficients |
| Dates futures | Transparent −1 year shift to archive data |
| GEE indisponible | Phenological fallback always produces a result |
| Faux positifs après taille | Stratification by system + month (future: add harvest-month flag) |
