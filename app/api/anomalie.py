from fastapi import APIRouter

from app.models.anomalie import AnomalieRequest, AnomalieResponse
from app.services.sentinel import fetch_sentinel2
from app.services.weather import fetch_weather

router = APIRouter()

_BASELINES = {
    "extensif": [0.42, 0.44, 0.46, 0.45, 0.44],
    "intensif": [0.48, 0.50, 0.52, 0.51, 0.50],
    "hyper_intensif": [0.58, 0.61, 0.63, 0.62, 0.61],
}


def _centroid(polygone: dict) -> tuple[float, float]:
    coords = polygone.get("coordinates", [[]])[0]
    if not coords:
        return (36.8, 10.1)
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def _anomaly_score(observed: list[float], expected: list[float]) -> float:
    """Mean normalised deviation over the sliding window (unbounded positive float)."""
    pairs = zip(observed, expected)
    deviations = [(e - o) / (e + 1e-8) for o, e in pairs if e > 0]
    return (
        round(max(0.0, sum(deviations) / len(deviations)) * 10, 2)
        if deviations
        else 0.0
    )


def _status(score: float) -> str:
    if score < 1.0:
        return "vert"
    if score < 2.0:
        return "orange"
    return "rouge"


@router.post("/diagnostic-anomalie", response_model=AnomalieResponse)
async def diagnostic_anomalie(req: AnomalieRequest) -> AnomalieResponse:
    lat, lon = _centroid(req.oliveraie.polygone)
    bands = await fetch_sentinel2(req.oliveraie.polygone, req.date)
    weather = await fetch_weather(lat, lon, req.date)
    _ = bands, weather  # used by real baseline model

    # --- stub: replace with Ridge/Prophet baseline conditioned on phenology + weather ---
    ndvi_attendu = _BASELINES[req.oliveraie.systeme]
    ndvi_observe = [0.42, 0.45, 0.41, 0.37, 0.34]  # stub observed series

    score = _anomaly_score(ndvi_observe, ndvi_attendu)
    statut = _status(score)

    pct_below = round((1 - sum(ndvi_observe) / sum(ndvi_attendu)) * 100)
    explication = (
        f"NDVI {pct_below}% en dessous attendu sur 3 semaines, malgré pluie normale. "
        "Stress probable - vérifier irrigation."
        if statut != "vert"
        else "NDVI conforme aux attentes saisonnières pour ce système de culture."
    )

    recommandation = {
        "vert": "Aucune action requise. Continuer la surveillance hebdomadaire.",
        "orange": "Inspection visuelle dans 48h",
        "rouge": "Intervention urgente — stress hydrique ou phytosanitaire sévère détecté.",
    }[statut]

    return AnomalieResponse(
        statut=statut,
        anomaly_score=score,
        ndvi_observe=ndvi_observe,
        ndvi_attendu=ndvi_attendu,
        explication=explication,
        recommandation=recommandation,
    )
