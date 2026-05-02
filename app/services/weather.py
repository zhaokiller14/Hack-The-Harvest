"""Weather data retrieval via Open-Meteo Archive/Forecast APIs (no API key required)."""
import asyncio
from datetime import date, timedelta
from typing import Any

import httpx

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_DAILY_VARS = (
    "precipitation_sum,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration"
)
_TIMEOUT = 15.0


def _aggregate(daily: dict[str, list]) -> dict[str, Any]:
    """Aggregate daily arrays into 21-day summary features."""
    rain = [v or 0.0 for v in daily.get("precipitation_sum", [])]
    tmax = [v or 20.0 for v in daily.get("temperature_2m_max", [])]
    tmin = [v or 10.0 for v in daily.get("temperature_2m_min", [])]
    et0 = [v or 0.0 for v in daily.get("et0_fao_evapotranspiration", [])]

    gdd = [max(0.0, (hi + lo) / 2.0 - 10.0) for hi, lo in zip(tmax, tmin)]

    return {
        "rainfall_mm": round(sum(rain), 1),
        "et0_mm": round(sum(et0), 1),
        "gdd": round(sum(gdd), 1),
        "heat_stress_days": sum(1 for t in tmax if t > 35),
        "_daily_rain": rain,
        "_daily_tmax": tmax,
        "_daily_et0": et0,
    }


def _tunisia_seasonal_fallback(center_date: date) -> dict[str, Any]:
    """Return typical Tunisian weather stats when the API is unavailable."""
    m = center_date.month
    # Rough monthly normals (Cap Bon / Sfax blend)
    rain_monthly = [50, 40, 30, 20, 12, 3, 1, 2, 20, 40, 55, 50]
    et0_monthly = [58, 63, 78, 98, 128, 152, 168, 162, 128, 93, 63, 53]
    tmax_monthly = [15, 17, 20, 24, 29, 34, 37, 37, 32, 27, 21, 16]
    tmin_monthly = [7, 8, 10, 13, 17, 21, 24, 24, 20, 16, 11, 8]

    idx = m - 1
    rain_21d = rain_monthly[idx] * 21 / 30
    et0_21d = et0_monthly[idx] * 21 / 30
    t_hi = tmax_monthly[idx]
    t_lo = tmin_monthly[idx]
    gdd = max(0.0, (t_hi + t_lo) / 2.0 - 10.0) * 21
    heat = max(0, int((t_hi - 35) * 2)) if t_hi > 35 else 0

    return {
        "rainfall_mm": round(rain_21d, 1),
        "et0_mm": round(et0_21d, 1),
        "gdd": round(gdd, 1),
        "heat_stress_days": heat,
        "_daily_rain": [rain_21d / 21] * 21,
        "_daily_tmax": [float(t_hi)] * 21,
        "_daily_et0": [et0_21d / 21] * 21,
    }


async def fetch_weather(lat: float, lon: float, date_str: str, window_days: int = 21) -> dict[str, Any]:
    """
    Fetch aggregated weather for a `window_days`-day window ending on `date_str`.

    Automatically routes to:
    - Archive API  for past dates
    - Forecast API for dates up to +16 days ahead
    - Last-year proxy for dates further in the future
    """
    end = date.fromisoformat(date_str)
    today = date.today()

    # For future dates beyond forecast horizon, shift back 1 year as climate proxy
    if end > today + timedelta(days=16):
        end = end.replace(year=end.year - 1)

    start = end - timedelta(days=window_days - 1)

    # Route to correct API
    if end <= today:
        url = _ARCHIVE_URL
    else:
        url = _FORECAST_URL

    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": _DAILY_VARS,
        "timezone": "Africa/Tunis",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return _aggregate(data.get("daily", {}))
    except Exception:
        center = start + timedelta(days=window_days // 2)
        return _tunisia_seasonal_fallback(center)


async def fetch_weather_series(
    lat: float,
    lon: float,
    end_date_str: str,
    n_steps: int = 5,
    step_days: int = 14,
) -> list[dict[str, Any]]:
    """
    Fetch weather for `n_steps` bi-weekly windows ending at `end_date_str`.
    Returns list ordered oldest → newest (index 0 = oldest window).
    """
    end = date.fromisoformat(end_date_str)
    today = date.today()
    if end > today + timedelta(days=16):
        end = end.replace(year=end.year - 1)

    tasks = []
    for i in range(n_steps - 1, -1, -1):
        window_end = end - timedelta(days=i * step_days)
        tasks.append(fetch_weather(lat, lon, window_end.isoformat()))

    return list(await asyncio.gather(*tasks))
