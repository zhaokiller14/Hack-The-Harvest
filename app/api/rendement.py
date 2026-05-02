from fastapi import APIRouter, HTTPException
import shapely.geometry as sg

from app.models.rendement import RendementRequest, RendementResponse, ShapFeature, IncertitudeInfo
from app.services.sentinel import fetch_sentinel2
from app.services.weather import fetch_weather
from app.services.soil import fetch_soil
from app.services.mlp_model import get_predictor

router = APIRouter()


def _centroid(polygone: dict) -> tuple[float, float]:
    coords = polygone.get("coordinates", [[]])[0]
    if not coords:
        return (36.8, 10.1)
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def _area_ha(polygone: dict) -> float:
    """Compute parcel area in hectares from a GeoJSON polygon (WGS84)."""
    try:
        geom = sg.shape(polygone)
        # Project to an equal-area CRS approximation using degree-to-metre ratio
        lat = geom.centroid.y
        import math
        m_per_deg_lat = 111_132.0
        m_per_deg_lon = 111_132.0 * math.cos(math.radians(lat))
        # Scale polygon to metres and compute area
        scaled = sg.Polygon([
            (x * m_per_deg_lon, y * m_per_deg_lat)
            for x, y in geom.exterior.coords
        ])
        return max(0.1, scaled.area / 10_000)  # m² → ha
    except Exception:
        return 2.5


@router.post("/predire-rendement", response_model=RendementResponse)
async def predire_rendement(req: RendementRequest) -> RendementResponse:
    try:
        lat, lon = _centroid(req.parcelle.polygone)
        area_ha  = _area_ha(req.parcelle.polygone)

        # Fetch all data sources in parallel
        import asyncio
        bands, weather, soil = await asyncio.gather(
            fetch_sentinel2(
                polygone=req.parcelle.polygone,
                date_prediction=req.date_prediction,
                date_plantation=req.parcelle.date_plantation,
            ),
            fetch_weather(lat, lon, req.date_prediction,
                          date_plantation=req.parcelle.date_plantation),
            fetch_soil(lat, lon),
        )

        predictor = get_predictor()
        result = predictor.predict(
            bands=bands,
            weather=weather,
            date_plantation=req.parcelle.date_plantation,
            date_prediction=req.date_prediction,
            soil=soil,
            area_ha=area_ha,
        )

        shap_features = [
            ShapFeature(feature=f["feature"], impact=f["impact"])
            for f in result["top_features_shap"]
        ]

        return RendementResponse(
            tonnage_predit_t=result["tonnage_predit_t"],
            intervalle_confiance_95=result["intervalle_confiance_95"],
            top_features_shap=shap_features,
            date_recolte_estimee=result["date_recolte_estimee"],
            incertitude=IncertitudeInfo(
                niveau=result["incertitude_niveau"],
                sigma_log=result["incertitude_sigma_log"]
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur prédiction rendement: {str(e)}"
        )
