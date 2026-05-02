"""Weather data retrieval via Open-Meteo (zero friction, no API key)."""
from typing import Any


async def fetch_weather(lat: float, lon: float, date: str) -> dict[str, Any]:
    """Fetch aggregated weather for a point location around a given date.

    Returns cumulative rainfall, heat stress days, ET0, GDD.
    Replace stub with real Open-Meteo / CHIRPS call.
    """
    # TODO: httpx call to https://api.open-meteo.com/v1/forecast
    return {
        "rainfall_mm": 42.0,
        "heat_stress_days": 3,
        "et0_mm": 5.2,
        "gdd": 180.0,
    }
