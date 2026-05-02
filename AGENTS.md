# Hack The Harvest

## Environment
- Python venv is at `.venv/` — always use `.venv/bin/python` and `.venv/bin/uvicorn`, not system `python`
- Run app: `.venv/bin/uvicorn app.main:app --reload`
- Pyright import errors for fastapi/pydantic are false positives (packages are in `.venv`, not system site-packages)

## Project Structure
- `app/main.py` — FastAPI entry point, CORS, router registration
- `app/api/` — one file per track (cartographier, rendement, anomalie)
- `app/models/` — Pydantic request/response schemas per track
- `app/services/` — shared stubs: sentinel.py, ndvi.py, weather.py

## API Contracts (must match exactly)
- Track 1 `POST /api/cartographier`: oliveraies items are flat (`polygone`, `systeme`, `confiance`, `surface_ha`) — not nested under `properties`
- Track 2 `POST /api/predire-rendement`: `intervalle_confiance_95` is a 2-element `list[float]`, not a dict; SHAP feature field is `impact` (signed float), not `importance`/`direction`
- Track 3 `POST /api/diagnostic-anomalie`: `ndvi_observe` and `ndvi_attendu` are `list[float]` time series; `anomaly_score` is unbounded (e.g. 2.4), not clamped 0–1

## Hackathon Tracks
- Track 1: Olive grove mapping — U-Net segmentation + RF/XGBoost classification on Sentinel-2
- Track 2: Tomato yield prediction — LightGBM/XGBoost + SHAP + conformal intervals
- Track 3: Olive anomaly detection — Ridge/Prophet NDVI baseline vs. observed
- Track 4: Multimodal Darija assistant — separate app (Whisper + CNN + RAG + TTS), not in this FastAPI
