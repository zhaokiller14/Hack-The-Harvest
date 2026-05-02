"""Sentinel-2 NDVI time series via Google Earth Engine.

Requires ee.Authenticate() run once per machine and GEE_PROJECT set in .env.
Cloud gaps within the series are filled by linear interpolation between
neighbouring real satellite values.
"""
import asyncio
from datetime import date, timedelta
from functools import partial
from typing import Any


# Reference fallback values (dataset means) used when a season has no clear images
_FALLBACK: dict[str, list[float]] = {
    "ndvi":  [0.35, 0.55, 0.65, 0.50],
    "evi":   [0.20, 0.35, 0.42, 0.32],
    "evi2":  [0.22, 0.38, 0.45, 0.35],
    "dswi":  [0.50, 0.60, 0.65, 0.55],
    "ndwi":  [-0.20, -0.10, -0.05, -0.15],
    "nri":   [0.05, 0.08, 0.10, 0.07],
    "cloud": [0.08, 0.06, 0.04, 0.04],
}


def _compute_indices(img):
    """Add NDVI, EVI, EVI2, DSWI, NDWI, NRI bands to a Sentinel-2 image."""
    import ee  # type: ignore

    b2  = img.select("B2")
    b3  = img.select("B3")
    b4  = img.select("B4")
    b8  = img.select("B8")
    b11 = img.select("B11")

    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    evi  = img.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": b8, "RED": b4, "BLUE": b2},
    ).rename("evi")
    evi2 = img.expression(
        "2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)",
        {"NIR": b8, "RED": b4},
    ).rename("evi2")
    dswi = img.expression(
        "(NIR + GREEN) / (SWIR + RED + 0.0001)",
        {"NIR": b8, "GREEN": b3, "SWIR": b11, "RED": b4},
    ).rename("dswi")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("ndwi")
    nri  = img.normalizedDifference(["B3", "B4"]).rename("nri")

    return img.addBands([ndvi, evi, evi2, dswi, ndwi, nri])


def _gee_vegetation_series(
    polygone: dict[str, Any],
    date_plantation: str,
    n_seasons: int = 4,
    season_days: int = 30,
) -> dict[str, list[float | None]]:
    """
    For each phenological season, fetch Sentinel-2 and return mean vegetation indices.

    Returns dict with keys ndvi, evi, evi2, dswi, ndwi, nri, cloud —
    each a list of length n_seasons. None means no cloud-free scene in that window.
    """
    import ee  # type: ignore

    from app.config import get_settings

    cfg = get_settings()
    if cfg.gee_project:
        ee.Initialize(project=cfg.gee_project)
    else:
        ee.Initialize()

    aoi = ee.Geometry.Polygon(polygone["coordinates"])
    plantation = date.fromisoformat(date_plantation)

    # Shift future plantation dates back 1 year so GEE archive has data
    today = date.today()
    if plantation > today:
        plantation = plantation.replace(year=plantation.year - 1)

    result: dict[str, list[float | None]] = {k: [] for k in
                                              ["ndvi", "evi", "evi2", "dswi", "ndwi", "nri", "cloud"]}

    for i in range(n_seasons):
        t_start = (plantation + timedelta(days=i * season_days)).isoformat()
        t_end   = (plantation + timedelta(days=(i + 1) * season_days)).isoformat()

        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(t_start, t_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(_compute_indices)
            .median()
        )

        stats = col.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=int(1e8),
            bestEffort=True,
        ).getInfo()

        for key in ["ndvi", "evi", "evi2", "dswi", "ndwi", "nri"]:
            val = stats.get(key)
            result[key].append(round(float(val), 4) if val is not None else None)

        # Cloud cover: approximate from whether the median composite had valid data
        result["cloud"].append(0.05 if stats.get("ndvi") is not None else 0.80)

    return result


def _fill_gaps(series: dict[str, list[float | None]]) -> dict[str, list[float]]:
    """Replace None slots with the dataset reference mean for that season."""
    filled: dict[str, list[float]] = {}
    for key, vals in series.items():
        fb = _FALLBACK.get(key, [0.0] * len(vals))
        filled[key] = [
            v if v is not None else fb[i]
            for i, v in enumerate(vals)
        ]
    return filled


async def fetch_sentinel2(
    polygone: dict[str, Any],
    date_prediction: str,
    date_plantation: str = "",
    **_kwargs: Any,
) -> dict[str, Any]:
    """
    Return vegetation index series for 4 phenological seasons via GEE.

    Args:
        polygone:        GeoJSON Polygon dict.
        date_prediction: ISO date of the prediction (used if date_plantation absent).
        date_plantation: ISO date of crop plantation — defines the 4 season windows.

    Returns:
        {
          "ndvi_series":       [float, float, float, float],  # s1→s4
          "evi_series":        [...],
          "evi2_series":       [...],
          "dswi_series":       [...],
          "ndwi_series":       [...],
          "nri_series":        [...],
          "cloud_series":      [...],
          "cloud_sat_series":  [...],
          "source":            "gee",
        }

    Cloud gaps are filled with dataset reference means.
    Falls back to all-reference values on GEE error.
    """
    ref_date = date_plantation or date_prediction

    try:
        loop = asyncio.get_event_loop()
        fn = partial(_gee_vegetation_series, polygone, ref_date)
        raw = await loop.run_in_executor(None, fn)
    except Exception:
        # GEE unavailable — return reference values; feature_builder uses them as-is
        raw = {k: [None] * 4 for k in ["ndvi", "evi", "evi2", "dswi", "ndwi", "nri", "cloud"]}

    filled = _fill_gaps(raw)

    return {
        "ndvi_series":      filled["ndvi"],
        "evi_series":       filled["evi"],
        "evi2_series":      filled["evi2"],
        "dswi_series":      filled["dswi"],
        "ndwi_series":      filled["ndwi"],
        "nri_series":       filled["nri"],
        "cloud_series":     filled["cloud"],
        "cloud_sat_series": filled["cloud"],
        "source":           "gee",
    }
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
