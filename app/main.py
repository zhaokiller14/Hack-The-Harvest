from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import anomalie, cartographier, rendement

app = FastAPI(
    title="Hack The Harvest API",
    description="Olive grove mapping, tomato yield prediction, and anomaly detection.",
    version="0.1.0",
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
