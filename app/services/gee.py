"""Google Earth Engine feature extraction for a single parcel polygon."""
import asyncio
from functools import partial
from typing import Any

from app.config import get_settings


def _extract_sync(polygon_geojson: dict[str, Any], area_ha: float) -> dict[str, float]:
    import ee  # type: ignore

    cfg = get_settings()
    ee.Initialize(project=cfg.gee_project)
    aoi = ee.Geometry.Polygon(polygon_geojson["coordinates"])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(cfg.sentinel_date_start, cfg.sentinel_date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.sentinel_cloud_pct))
        .median()
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndwi = s2.normalizedDifference(["B8", "B11"]).rename("ndwi")
    ndre = s2.normalizedDifference(["B8", "B5"]).rename("ndre")

    stack = ndvi.addBands(ndwi).addBands(ndre)
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True)
    )
    stats = stack.reduceRegion(
        reducer=reducer,
        geometry=aoi,
        scale=10,
        maxPixels=int(1e8),
        bestEffort=True,
    ).getInfo()

    def g(key: str) -> float:
        return float(stats.get(key) or 0.0)

    ndvi_mean = g("ndvi_mean")
    ndvi_std = g("ndvi_stdDev")
    ndwi_mean = g("ndwi_mean")

    return {
        "ndvi_mean": ndvi_mean,
        "ndvi_std": ndvi_std,
        "ndvi_p10": g("ndvi_p10"),
        "ndvi_p90": g("ndvi_p90"),
        "ndvi_amplitude": g("ndvi_p90") - g("ndvi_p10"),
        "ndwi_mean": ndwi_mean,
        "ndwi_std": g("ndwi_stdDev"),
        "ndre_mean": g("ndre_mean"),
        "ndre_std": g("ndre_stdDev"),
        "area_ha": area_ha,
        "ndvi_ndwi_ratio": ndvi_mean / (ndwi_mean + 1e-8),
        "texture_proxy": ndvi_std / (ndvi_mean + 1e-8),
    }


async def extract_features(polygon_geojson: dict[str, Any], area_ha: float) -> dict[str, float]:
    """Run GEE extraction in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    fn = partial(_extract_sync, polygon_geojson, area_ha)
    return await loop.run_in_executor(None, fn)
