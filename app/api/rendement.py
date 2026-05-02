from fastapi import APIRouter, HTTPException

from app.models.rendement import RendementRequest, RendementResponse, ShapFeature, IncertitudeInfo
from app.services.sentinel import fetch_sentinel2
from app.services.weather import fetch_weather
from app.services.mlp_model import get_predictor

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
    try:
        # Collecte des données externes
        lat, lon = _centroid(req.parcelle.polygone)
        bands = await fetch_sentinel2(req.parcelle.polygone, req.date_prediction)
        weather = await fetch_weather(lat, lon, req.date_prediction)

        # Prédiction avec MLP + MC Dropout
        predictor = get_predictor()
        result = predictor.predict(
            bands=bands,
            weather=weather, 
            date_plantation=req.parcelle.date_plantation,
            date_prediction=req.date_prediction
        )

        # Conversion des features SHAP
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
