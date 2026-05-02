"""Weather data retrieval via Open-Meteo archive API (no API key required)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import httpx

_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
_DAILY_VARS = ",".join([
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
])
_GDD_BASE = 10.0        # base temperature for tomato GDD (°C)
_DROUGHT_PRECIP = 2.0   # mm/day threshold for "dry day"
_HEAT_TMAX = 32.0       # °C threshold for heat stress day


def _season_windows(date_plantation: str) -> list[tuple[datetime, datetime]]:
    """Return (start, end) datetime pairs for the 4 phenological seasons."""
    p = datetime.strptime(date_plantation, "%Y-%m-%d")
    return [(p + timedelta(days=30 * i), p + timedelta(days=30 * (i + 1)))
            for i in range(4)]


def _aggregate(daily: dict[str, list], start: datetime, end: datetime) -> dict[str, float]:
    """Aggregate Open-Meteo daily arrays for [start, end) window."""
    times = [datetime.strptime(t, "%Y-%m-%d") for t in daily["time"]]
    idx = [i for i, t in enumerate(times) if start <= t < end]
    if not idx:
        return {}

    def pick(key: str) -> list[float]:
        vals = daily.get(key, [])
        return [vals[i] for i in idx if i < len(vals) and vals[i] is not None]

    tmax  = pick("temperature_2m_max")
    tmin  = pick("temperature_2m_min")
    tmean = pick("temperature_2m_mean")
    precip = pick("precipitation_sum")
    et0   = pick("et0_fao_evapotranspiration")
    solar = pick("shortwave_radiation_sum")
    wind  = pick("windspeed_10m_max")
    hmax  = pick("relative_humidity_2m_max")
    hmin  = pick("relative_humidity_2m_min")
    hmean = pick("relative_humidity_2m_mean")

    n = max(len(tmax), 1)

    gdd_vals = [max(0.0, (tx + tn) / 2 - _GDD_BASE) for tx, tn in zip(tmax, tmin)]

    precip_cum  = sum(precip)
    et0_sum     = sum(et0)
    stress_hyd  = max(0.0, et0_sum - precip_cum)

    return {
        "gdd":                  sum(gdd_vals),
        "temp_mean":            sum(tmean) / n if tmean else 20.0,
        "temp_max":             max(tmax) if tmax else 28.0,
        "temp_min":             min(tmin) if tmin else 12.0,
        "temp_amplitude":       (max(tmax) - min(tmin)) if (tmax and tmin) else 12.0,
        "precip_cum":           precip_cum,
        "et0_mean":             sum(et0) / n if et0 else 5.0,
        "solar_cum":            sum(solar),
        "wind_speed_max":       max(wind) if wind else 25.0,
        "humidity_mean":        sum(hmean) / n if hmean else 60.0,
        "humidity_max":         max(hmax) if hmax else 90.0,
        "humidity_min":         min(hmin) if hmin else 30.0,
        "freq_secheresse":      sum(1 for p in precip if p < _DROUGHT_PRECIP),
        "intensite_chaleur":    sum(1 for t in tmax if t > _HEAT_TMAX),
        "stress_hydrique":      stress_hyd,
        "intensite_stress":     stress_hyd / n,
        "precipitation_hours":  sum(1 for p in precip if p > 0),
        # soil_moisture proxy: normalized water balance (precipitation / et0)
        "soil_moisture":        precip_cum / et0_sum if et0_sum > 0 else 0.5,
        # soil_temp proxy: mean air temp offset
        "soil_temp":            (sum(tmean) / n - 2.0) if tmean else 18.0,
    }


async def fetch_weather(lat: float, lon: float, date_prediction: str,
                        date_plantation: str = "") -> dict[str, Any]:
    """
    Fetch aggregated weather for all 4 phenological seasons.

    Returns nested dict: weather["s1"]["gdd"], weather["s1"]["temp_mean"], …
    Falls back to climatological defaults for Tunisia on any error.
    """
    seasons = _season_windows(date_plantation or date_prediction)
    overall_start = seasons[0][0]
    overall_end   = seasons[-1][1]

    # Clamp end to yesterday (archive API doesn't have future data)
    yesterday = datetime.utcnow() - timedelta(days=1)
    overall_end = min(overall_end, yesterday)
    if overall_start >= overall_end:
        return _fallback_weather()

    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": overall_start.strftime("%Y-%m-%d"),
        "end_date":   overall_end.strftime("%Y-%m-%d"),
        "daily":      _DAILY_VARS,
        "timezone":   "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(_BASE_URL, params=params)
            r.raise_for_status()
            daily = r.json()["daily"]
    except Exception:
        return _fallback_weather()

    result: dict[str, Any] = {}
    for i, (start, end) in enumerate(seasons):
        season_key = f"s{i + 1}"
        end_clamped = min(end, yesterday)
        result[season_key] = _aggregate(daily, start, end_clamped)

    # Season totals (used for static features)
    result["total"] = {
        "gdd_total":    sum(result[f"s{i+1}"].get("gdd", 0)         for i in range(4)),
        "precip_total": sum(result[f"s{i+1}"].get("precip_cum", 0)  for i in range(4)),
        "stress_total": sum(result[f"s{i+1}"].get("stress_hydrique", 0) for i in range(4)),
        "ndvi_max_saison": 0.0,   # filled by sentinel
    }

    return result


def _fallback_weather() -> dict[str, Any]:
    """Climatological defaults for central Tunisia tomato season."""
    defaults = [
        {"gdd": 120, "temp_mean": 14, "temp_max": 19, "temp_min": 9,
         "temp_amplitude": 10, "precip_cum": 42, "et0_mean": 3.2,
         "solar_cum": 155, "wind_speed_max": 28, "humidity_mean": 75,
         "humidity_max": 90, "humidity_min": 55, "freq_secheresse": 5,
         "intensite_chaleur": 0, "stress_hydrique": 54, "intensite_stress": 1.8,
         "precipitation_hours": 8, "soil_moisture": 0.78, "soil_temp": 12},
        {"gdd": 260, "temp_mean": 19, "temp_max": 24, "temp_min": 14,
         "temp_amplitude": 10, "precip_cum": 30, "et0_mean": 4.8,
         "solar_cum": 210, "wind_speed_max": 28, "humidity_mean": 62,
         "humidity_max": 82, "humidity_min": 42, "freq_secheresse": 10,
         "intensite_chaleur": 1, "stress_hydrique": 114, "intensite_stress": 3.8,
         "precipitation_hours": 5, "soil_moisture": 0.47, "soil_temp": 17},
        {"gdd": 400, "temp_mean": 24, "temp_max": 30, "temp_min": 18,
         "temp_amplitude": 12, "precip_cum": 15, "et0_mean": 6.5,
         "solar_cum": 270, "wind_speed_max": 30, "humidity_mean": 50,
         "humidity_max": 72, "humidity_min": 28, "freq_secheresse": 22,
         "intensite_chaleur": 6, "stress_hydrique": 180, "intensite_stress": 6.0,
         "precipitation_hours": 3, "soil_moisture": 0.23, "soil_temp": 22},
        {"gdd": 510, "temp_mean": 28, "temp_max": 35, "temp_min": 21,
         "temp_amplitude": 14, "precip_cum": 8,  "et0_mean": 8.0,
         "solar_cum": 310, "wind_speed_max": 32, "humidity_mean": 42,
         "humidity_max": 65, "humidity_min": 20, "freq_secheresse": 27,
         "intensite_chaleur": 14, "stress_hydrique": 232, "intensite_stress": 7.7,
         "precipitation_hours": 1, "soil_moisture": 0.10, "soil_temp": 26},
    ]
    result = {f"s{i+1}": d for i, d in enumerate(defaults)}
    result["total"] = {
        "gdd_total":    sum(d["gdd"]           for d in defaults),
        "precip_total": sum(d["precip_cum"]    for d in defaults),
        "stress_total": sum(d["stress_hydrique"] for d in defaults),
        "ndvi_max_saison": 0.0,
    }
    return result
