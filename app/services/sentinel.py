"""Sentinel-2 NDVI time series via Google Earth Engine.

Requires ee.Authenticate() run once per machine and GEE_PROJECT set in .env.
Cloud gaps within the series are filled by linear interpolation between
neighbouring real satellite values.
"""
import asyncio
from datetime import date, timedelta
from functools import partial
from typing import Any


def _gee_ndvi_series(
    polygone: dict[str, Any],
    end_date_str: str,
    n_steps: int,
    step_days: int,
) -> list[float | None]:
    """Return per-step mean NDVI from Sentinel-2 via GEE.

    Returns None for steps where no cloud-free scene exists in the window.
    """
    import ee  # type: ignore  # noqa: PLC0415

    from app.config import get_settings

    cfg = get_settings()
    if cfg.gee_project:
        ee.Initialize(project=cfg.gee_project)
    else:
        ee.Initialize()

    aoi = ee.Geometry.Polygon(polygone["coordinates"])
    end = date.fromisoformat(end_date_str)
    # For future dates, shift back 1 year so GEE archive has data
    today = date.today()
    if end > today:
        end = end.replace(year=end.year - 1)

    series: list[float | None] = []

    for i in range(n_steps - 1, -1, -1):
        obs_date = end - timedelta(days=i * step_days)
        t_start = (obs_date - timedelta(days=7)).isoformat()
        t_end = obs_date.isoformat()

        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(t_start, t_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("ndvi"))
            .median()
        )
        stats = col.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=int(1e8),
            bestEffort=True,
        ).getInfo()
        val = stats.get("ndvi")
        series.append(round(float(val), 4) if val is not None else None)

    return series


def _interpolate_gaps(series: list[float | None]) -> list[float]:
    """Fill None slots by linear interpolation between neighbouring real values."""
    result = list(series)
    n = len(result)

    # Forward-fill leading Nones from the first real value
    first_real = next((i for i, v in enumerate(result) if v is not None), None)
    if first_real is None:
        raise ValueError("No valid NDVI observations in the series (all steps cloudy).")
    for i in range(first_real):
        result[i] = result[first_real]

    # Interpolate interior Nones
    i = 0
    while i < n:
        if result[i] is None:
            left = i - 1
            right = next((j for j in range(i + 1, n) if result[j] is not None), None)
            if right is None:
                # Trailing Nones — back-fill from last real value
                for j in range(i, n):
                    result[j] = result[left]
                break
            for j in range(i, right):
                t = (j - left) / (right - left)
                result[j] = round(result[left] + t * (result[right] - result[left]), 4)
            i = right
        else:
            i += 1

    return result  # type: ignore[return-value]


async def fetch_sentinel2(
    polygone: dict[str, Any],
    date_str: str,
    n_steps: int = 5,
    step_days: int = 14,
    **_kwargs: Any,
) -> dict[str, Any]:
    """
    Return NDVI time series for the parcel via GEE (n_steps bi-weekly values, oldest→newest).

    Cloud-gap steps are filled by linear interpolation between real satellite values.

    Args:
        polygone:  GeoJSON Polygon dict.
        date_str:  ISO date of the most recent observation.
        n_steps:   Number of time steps in the series.
        step_days: Days between steps.

    Returns:
        {"ndvi_series": [float, ...], "source": "gee"}

    Raises:
        RuntimeError if GEE is not authenticated or all steps are cloudy.
    """
    loop = asyncio.get_event_loop()
    fn = partial(_gee_ndvi_series, polygone, date_str, n_steps, step_days)
    raw_series: list[float | None] = await loop.run_in_executor(None, fn)

    ndvi_series = _interpolate_gaps(raw_series)
    return {"ndvi_series": ndvi_series, "source": "gee"}
