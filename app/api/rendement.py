from fastapi import APIRouter

from app.models.rendement import RendementRequest, RendementResponse, ShapFeature
from app.services.sentinel import fetch_sentinel2
from app.services.weather import fetch_weather

router = APIRouter()


def _centroid(polygone: dict) -> tuple[float, float]:
    coords = polygone.get("coordinates", [[]])[0]
    if not coords:
        return (36.8, 10.1)
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


@router.post("/predire-rendement", response_model=RendementResponse)
async def predire_rendement(req: RendementRequest) -> RendementResponse:
    lat, lon = _centroid(req.parcelle.polygone)
    bands = await fetch_sentinel2(req.parcelle.polygone, req.date_prediction)
    weather = await fetch_weather(lat, lon, req.date_prediction)

    # --- stub: replace with LightGBM/XGBoost model + SHAP explainer ---
    _ = bands, weather

    return RendementResponse(
        tonnage_predit_t=52.3,
        intervalle_confiance_95=[46.1, 58.7],
        top_features_shap=[
            ShapFeature(feature="NDVI_max_juillet", impact=8.4),
            ShapFeature(feature="cumul_pluie_juin", impact=-3.1),
            ShapFeature(feature="jours_stress_thermique", impact=-2.7),
        ],
        date_recolte_estimee="2026-09-22",
    )
