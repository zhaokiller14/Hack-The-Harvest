"""Track 1 — Olive grove cartography pipeline."""

import asyncio
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import shapely.geometry as sg
from shapely import STRtree
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.cartographier import (
    CartographierRequest,
    CartographierResponse,
    CartographierStats,
    OliveraieFeature,
)
from app.services import classifier as clf_svc
from app.services import gee
from app.services import segmentation as seg_svc

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory state (loaded at startup via lifespan in main.py)
# ---------------------------------------------------------------------------
_parcels: list[dict[str, Any]] = []  # list of GeoJSON features
_parcel_shapes: list = []              # shapely geometries for spatial index
_strtree: STRtree | None = None       # spatial index
_feature_cache: dict[str, dict] = {}  # parcel_id → pre-extracted features
_job_store: dict[str, dict] = {}  # job_id → full GeoJSON result

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_parcels_and_cache() -> None:
    global _parcels, _parcel_shapes, _strtree, _feature_cache

    labeled_path = _DATA_DIR / "parcels_labeled.geojson"
    cache_path = _DATA_DIR / "feature_cache.json"

    if labeled_path.exists():
        geojson = json.loads(labeled_path.read_text())
        _parcels = geojson["features"]
        _parcel_shapes = [sg.shape(f["geometry"]) for f in _parcels]
        _strtree = STRtree(_parcel_shapes)

    if cache_path.exists():
        _feature_cache = json.loads(cache_path.read_text())

    clf_svc.load_model()
    seg_svc.load_unet_model()

    print(
        f"[cartographier] Loaded {len(_parcels)} EZZAYRA parcels, "
        f"{len(_feature_cache)} cached, model={'yes' if clf_svc.is_loaded() else 'no'}, "
        f"unet={'yes' if seg_svc.is_loaded() else 'no (OSM fallback)'}"
    )


# ---------------------------------------------------------------------------
# OSM Overpass fallback for unseen zones
# ---------------------------------------------------------------------------
async def _fetch_osm_orchards(polygon_geojson: dict[str, Any]) -> list[dict[str, Any]]:
    """Query Overpass API for landuse=orchard within the polygon bbox."""
    import httpx

    coords = polygon_geojson["coordinates"][0]
    lats = [c[1] for c in coords]
    lngs = [c[0] for c in coords]
    south, north = min(lats), max(lats)
    west, east = min(lngs), max(lngs)

    query = f"""
    [out:json][timeout:25];
    (
      way["landuse"="orchard"]({south},{west},{north},{east});
      relation["landuse"="orchard"]({south},{west},{north},{east});
    );
    out geom;
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://overpass-api.de/api/interpreter", data=query
            )
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
    except Exception:
        return []

    features = []
    for el in elements:
        if "geometry" not in el:
            continue
        ring = [[n["lon"], n["lat"]] for n in el["geometry"]]
        if len(ring) < 3:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        poly = {"type": "Polygon", "coordinates": [ring]}
        shape = sg.shape(poly)
        features.append(
            {
                "type": "Feature",
                "geometry": poly,
                "properties": {
                    "id": f"osm_{el['id']}",
                    "systeme": None,
                    "area_ha": shape.area * 1e-4 * 111320**2 * 0.85,  # rough ha
                    "source": "osm",
                },
            }
        )
    return features


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@router.post("/cartographier", response_model=CartographierResponse)
async def cartographier(req: CartographierRequest) -> CartographierResponse:
    drawn_shape = sg.shape(req.polygone_perimetre)

    # --- Stage 1: find intersecting EZZAYRA parcels ---
    matched: list[dict[str, Any]] = []
    if _strtree is not None:
        candidate_idxs = _strtree.query(drawn_shape, predicate="intersects")
        matched = [_parcels[i] for i in candidate_idxs]

    # --- Unknown zone: U-Net segmentation, then OSM fallback ---
    if not matched:
        if seg_svc.is_loaded():
            matched = await seg_svc.segment_polygon(req.polygone_perimetre)
        if not matched:
            matched = await _fetch_osm_orchards(req.polygone_perimetre)

    if not matched:
        return CartographierResponse(
            oliveraies=[],
            stats=CartographierStats(
                total_oliveraies=0,
                surface_totale_ha=0.0,
                repartition={"extensif": 0, "intensif": 0},
            ),
        )

    # --- Stage 2: classify each parcel (use cache or live GEE) ---
    async def classify_parcel(feat: dict[str, Any]) -> OliveraieFeature | None:
        parcel_id = feat["properties"].get("id", "")
        area_ha = feat["properties"].get("area_ha", 0.0)
        geometry = feat["geometry"]

        # Use pre-extracted cache if available
        if parcel_id in _feature_cache:
            features = dict(_feature_cache[parcel_id])
            features["area_ha"] = area_ha
        else:
            try:
                features = await gee.extract_features(geometry, area_ha)
            except Exception:
                features = {"area_ha": area_ha, "ndvi_mean": 0.5}

        systeme, confiance = clf_svc.predict(features)

        return OliveraieFeature(
            polygone=geometry,
            systeme=systeme,
            confiance=round(confiance, 3),
            surface_ha=round(area_ha, 2),
        )

    results = await asyncio.gather(*[classify_parcel(f) for f in matched])
    oliveraies: list[OliveraieFeature] = [r for r in results if r is not None]

    # --- Stats ---
    repartition: dict[str, int] = {"extensif": 0, "intensif": 0}
    for o in oliveraies:
        repartition[o.systeme] = repartition.get(o.systeme, 0) + 1

    stats = CartographierStats(
        total_oliveraies=len(oliveraies),
        surface_totale_ha=round(sum(o.surface_ha for o in oliveraies), 2),
        repartition=repartition,
    )

    # --- Store job for GeoJSON export ---
    job_id = _store_job(oliveraies, req.polygone_perimetre)

    response = CartographierResponse(oliveraies=oliveraies, stats=stats)
    # Attach job_id via response header (frontend uses it for download link)
    # We piggyback it as a non-schema field via a custom response when needed;
    # for now the export endpoint accepts a job_id query param.
    _ = job_id
    return response


def _store_job(oliveraies: list[OliveraieFeature], drawn_polygon: dict) -> str:
    job_id = str(uuid.uuid4())
    features = [
        {
            "type": "Feature",
            "geometry": o.polygone,
            "properties": {
                "systeme": o.systeme,
                "confiance": o.confiance,
                "surface_ha": o.surface_ha,
            },
        }
        for o in oliveraies
    ]
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {"drawn_polygon": drawn_polygon},
    }
    _job_store[job_id] = geojson
    # Also index by polygon hash so the frontend can retrieve without knowing job_id
    poly_hash = hashlib.md5(
        json.dumps(drawn_polygon, sort_keys=True).encode()
    ).hexdigest()
    _job_store[poly_hash] = geojson
    return job_id


# ---------------------------------------------------------------------------
# GeoJSON export endpoint
# ---------------------------------------------------------------------------
@router.get("/cartographier/export/{job_id}")
async def export_geojson(job_id: str):
    result = _job_store.get(job_id)
    if result is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(
        content=result,
        headers={
            "Content-Disposition": f'attachment; filename="oliveraies_{job_id[:8]}.geojson"'
        },
    )


@router.get("/cartographier/metrics")
async def unet_metrics():
    """Return U-Net segmentation test metrics."""
    metrics_path = Path(__file__).parent.parent.parent / "models" / "unet_metrics.json"
    if not metrics_path.exists():
        return JSONResponse({"error": "No metrics available yet"}, status_code=404)
    return JSONResponse(json.loads(metrics_path.read_text()))


@router.get("/cartographier/classifier-metrics")
async def classifier_metrics():
    """Return extensif/intensif classifier confusion matrix (test split)."""
    metrics_path = Path(__file__).parent.parent.parent / "models" / "classifier_metrics.json"
    if not metrics_path.exists():
        return JSONResponse({"error": "No classifier metrics available yet"}, status_code=404)
    return JSONResponse(json.loads(metrics_path.read_text()))
