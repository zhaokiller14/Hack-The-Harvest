"""Sentinel-2 L2A data retrieval (B2, B4, B8, B11)."""
from typing import Any


async def fetch_sentinel2(polygone: dict[str, Any], date: str) -> dict[str, Any]:
    """Download Sentinel-2 bands for the given GeoJSON polygon and date.

    Returns a dict keyed by band name with numpy-array-like data.
    Replace stub with real Copernicus / Sentinel Hub call.
    """
    # TODO: integrate sentinelhub-py or odc-stac
    return {
        "B02": [],
        "B04": [],
        "B08": [],
        "B11": [],
        "date": date,
        "crs": "EPSG:32632",
    }
