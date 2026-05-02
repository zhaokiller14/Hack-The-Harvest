from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import anomalie, cartographier, rendement
from app.api.cartographier import load_parcels_and_cache

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_parcels_and_cache()
    yield


app = FastAPI(
    title="Hack The Harvest API",
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

app.include_router(cartographier.router, prefix="/api", tags=["Track 1 — Cartographie"])
app.include_router(rendement.router, prefix="/api", tags=["Track 2 — Rendement"])
app.include_router(anomalie.router, prefix="/api", tags=["Track 3 — Anomalie"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")
