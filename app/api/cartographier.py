from fastapi import APIRouter

from app.models.cartographier import (
    CartographierRequest,
    CartographierResponse,
    CartographierStats,
    OliveraieFeature,
)
from app.services.ndvi import compute_ndvi
from app.services.sentinel import fetch_sentinel2

router = APIRouter()


@router.post("/cartographier", response_model=CartographierResponse)
async def cartographier(req: CartographierRequest) -> CartographierResponse:
    bands = await fetch_sentinel2(req.polygone_perimetre, req.date)
    ndvi = compute_ndvi(bands["B08"], bands["B04"])

    # --- stub: replace with real U-Net segmentation + RF/XGBoost classification ---
    oliveraies: list[OliveraieFeature] = [
        OliveraieFeature(
            polygone=req.polygone_perimetre,
            systeme="intensif",
            confiance=0.87,
            surface_ha=12.4,
        )
    ]

    stats = CartographierStats(
        total_oliveraies=47,
        surface_totale_ha=612.3,
        repartition={"extensif": 28, "intensif": 15, "hyper_intensif": 4},
    )

    return CartographierResponse(oliveraies=oliveraies, stats=stats)
