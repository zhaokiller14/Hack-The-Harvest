from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import anomalie, cartographier, rendement
from app.api.cartographier import load_parcels_and_cache

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
_PROPHET_DIR = Path(__file__).parent.parent / "models" / "prophet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_parcels_and_cache()
    yield


app = FastAPI(
    title="Ardhi API",
    description="Olive grove mapping, tomato yield prediction, and anomaly detection.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers — must be registered before the static-files catch-all
app.include_router(cartographier.router, prefix="/api", tags=["Track 1 — Cartographie"])
app.include_router(rendement.router,     prefix="/api", tags=["Track 2 — Rendement"])
app.include_router(anomalie.router,      prefix="/api", tags=["Track 3 — Anomalie"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/parcels")
async def list_parcels() -> dict:
    """List parcels that have a trained Prophet model, with geometry and systeme."""
    import json
    data_path = Path(__file__).parent.parent / "data" / "parcels_labeled.geojson"
    prophet_ids = {p.stem for p in _PROPHET_DIR.glob("*.json") if p.stem != "meta"}

    parcels = []
    if data_path.exists():
        fc = json.loads(data_path.read_text())
        for feat in fc.get("features", []):
            pid = feat["properties"].get("id", "")
            if pid in prophet_ids:
                parcels.append({
                    "id":       pid,
                    "systeme":  feat["properties"].get("systeme", "extensif"),
                    "area_ha":  feat["properties"].get("area_ha"),
                    "polygone": feat["geometry"],
                })
    else:
        parcels = [{"id": pid, "systeme": "extensif", "area_ha": None, "polygone": None}
                   for pid in sorted(prophet_ids)]

    parcels.sort(key=lambda p: p["id"])
    return {"parcels": parcels}


# Static files catch-all — serves index.html, rendement.html, anomalie.html, css/, js/
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
