"""POST /api/diagnostic-anomalie — olive grove anomaly detection."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import APIRouter

from app.models.anomalie import AnomalieRequest, AnomalieResponse
from app.services.parcel_baseline import (
    anomaly_zscore,
    get_expected_series,
    has_prophet_model,
)
from app.services.sentinel import fetch_sentinel2
from app.services.weather import fetch_weather_series

router = APIRouter()

# Thresholds: score < LOW → vert, < HIGH → orange, else rouge
_SCORE_LOW = 1.0
_SCORE_HIGH = 2.0


# ── Geometry helpers ───────────────────────────────────────────────────────

def _centroid(polygone: dict[str, Any]) -> tuple[float, float]:
    """Return (lat, lon) centroid of a GeoJSON Polygon."""
    coords = polygone.get("coordinates", [[]])[0]
    if not coords:
        return (36.8, 10.1)  # Tunisia centre fallback
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


# ── Status ─────────────────────────────────────────────────────────────────

def _status(score: float) -> str:
    if score < _SCORE_LOW:
        return "vert"
    if score < _SCORE_HIGH:
        return "orange"
    return "rouge"


# ── Explanation generation ─────────────────────────────────────────────────

def _explain(
    ndvi_observe: list[float],
    ndvi_attendu: list[float],
    score: float,
    statut: str,
    weather_series: list[dict],
    systeme: str,
) -> tuple[str, str]:
    obs_mean = sum(ndvi_observe) / len(ndvi_observe)
    exp_mean = sum(ndvi_attendu) / len(ndvi_attendu)
    pct_below = round((1 - obs_mean / (exp_mean + 1e-6)) * 100)

    last_w = weather_series[-1] if weather_series else {}
    rain = last_w.get("rainfall_mm", 20.0)
    et0 = last_w.get("et0_mm", 60.0)
    heat = last_w.get("heat_stress_days", 0)

    drought_index = max(0.0, et0 - rain) / (et0 + 1e-6)

    if drought_index > 0.7 and heat >= 5:
        weather_clause = "malgré pluie insuffisante et fortes chaleurs"
    elif drought_index > 0.7:
        weather_clause = "malgré déficit hydrique marqué"
    elif heat >= 5:
        weather_clause = f"avec {heat} jours de chaleur extrême"
    elif rain > 40:
        weather_clause = "malgré pluie normale"
    else:
        weather_clause = "dans des conditions météo normales"

    if statut == "vert":
        explication = (
            f"NDVI conforme aux attentes saisonnières pour un olivier {systeme}. "
            f"Écart observé: {abs(pct_below)}%."
        )
        recommandation = "Aucune action requise. Continuer la surveillance hebdomadaire."

    elif statut == "orange":
        explication = (
            f"NDVI {pct_below}% en dessous du attendu {weather_clause}. "
            f"Stress modéré détecté sur {systeme} — peut indiquer début de stress hydrique "
            f"ou attaque précoce de ravageurs."
        )
        recommandation = "Inspection visuelle dans 48h. Vérifier le système d'irrigation si applicable."

    else:  # rouge
        cause = "stress hydrique sévère ou problème phytosanitaire" if drought_index > 0.6 else "anomalie végétative critique"
        explication = (
            f"NDVI {pct_below}% en dessous attendu {weather_clause}. "
            f"{cause.capitalize()} détecté sur parcelle {systeme}. "
            f"Intervention urgente recommandée."
        )
        recommandation = (
            "Intervention urgente — inspecter l'irrigation, chercher ravageurs ou maladie foliaire. "
            "Contacter un technicien agricole sous 24h."
        )

    return explication, recommandation


# ── Main endpoint ──────────────────────────────────────────────────────────

@router.post("/diagnostic-anomalie", response_model=AnomalieResponse)
async def diagnostic_anomalie(req: AnomalieRequest) -> AnomalieResponse:
    lat, lon = _centroid(req.oliveraie.polygone)
    systeme = req.oliveraie.systeme

    # 1. Real weather (Open-Meteo Archive / Forecast API)
    weather_series = await fetch_weather_series(lat, lon, req.date, n_steps=5, step_days=14)

    # 2. Real NDVI time series from GEE
    ndvi_result = await fetch_sentinel2(
        polygone=req.oliveraie.polygone,
        date_str=req.date,
        n_steps=5,
        step_days=14,
    )
    ndvi_observe: list[float] = ndvi_result["ndvi_series"]

    # 3. Expected NDVI from Prophet model (per-parcel unsupervised baseline)
    if not has_prophet_model(req.oliveraie.id):
        raise ValueError(
            f"No Prophet model found for parcel '{req.oliveraie.id}'. "
            "Run scripts/build_ndvi_history.py then scripts/build_prophet_baselines.py first."
        )

    parcel_series = get_expected_series(req.oliveraie.id, req.date)
    ndvi_attendu = [round(mean, 4) for mean, _ in parcel_series]
    stds = [std for _, std in parcel_series]

    # 4. Z-score: deviation in units of Prophet's uncertainty interval
    score = anomaly_zscore(ndvi_observe, ndvi_attendu, stds)
    statut = _status(score)

    # 5. Explanation
    explication, recommandation = _explain(
        ndvi_observe, ndvi_attendu, score, statut, weather_series, systeme
    )

    return AnomalieResponse(
        statut=statut,
        anomaly_score=score,
        ndvi_observe=[round(v, 4) for v in ndvi_observe],
        ndvi_attendu=[round(v, 4) for v in ndvi_attendu],
        explication=explication,
        recommandation=recommandation,
    )
