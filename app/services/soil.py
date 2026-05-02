"""Soil properties retrieval via SoilGrids REST API (no API key required)."""

from __future__ import annotations

from typing import Any

import httpx

_BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
_PROPERTIES = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc"]
_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]

# d_factor maps SoilGrids mapped_units → target_units (divide raw by this)
_D_FACTOR: dict[str, float] = {
    "bdod":     100,   # cg/cm³ → kg/dm³ (= g/cm³)
    "cec":      10,    # mmol(c)/kg → cmol(c)/kg
    "cfvo":     10,    # cm³/dm³ → %
    "clay":     10,    # g/kg → %
    "nitrogen": 100,   # cg/kg → g/kg
    "phh2o":    10,    # pH × 10 → pH
    "sand":     10,    # g/kg → %
    "silt":     10,    # g/kg → %
    "soc":      10,    # dg/kg → g/kg
}

# Fallback: median values for Tunisian agricultural soils
_FALLBACK: dict[str, float] = {
    "bdod":     1.40,   # g/cm³
    "cec":      18.0,   # cmol/kg
    "cfvo":      5.0,   # %
    "clay":     28.0,   # %
    "nitrogen":  0.9,   # g/kg
    "phh2o":     7.6,   # pH
    "sand":     42.0,   # %
    "silt":     30.0,   # %
    "soc":      10.0,   # g/kg
}


async def fetch_soil(lat: float, lon: float) -> dict[str, float]:
    """
    Fetch topsoil (0-30 cm average) properties for the parcel centroid.

    Returns dict with keys: bdod, cec, cfvo, clay, nitrogen, phh2o, sand, silt, soc,
    awc, indice_texture_sol, ratio_sand_clay.
    Falls back to Tunisian defaults on error.
    """
    params: list[tuple[str, str]] = []
    for prop in _PROPERTIES:
        params.append(("property", prop))
    for depth in _DEPTHS:
        params.append(("depth", depth))
    params.append(("value", "mean"))
    params += [("lon", str(lon)), ("lat", str(lat))]

    raw: dict[str, float] = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(_BASE_URL, params=params)
            r.raise_for_status()
            layers = r.json()["properties"]["layers"]

        for layer in layers:
            name   = layer["name"]
            factor = _D_FACTOR.get(name, 1.0)
            vals   = [
                d["values"]["mean"]
                for d in layer["depths"]
                if d["values"]["mean"] is not None
            ]
            if vals:
                raw[name] = (sum(vals) / len(vals)) / factor
    except Exception:
        pass

    # Fill any missing properties with fallback
    soil = {k: raw.get(k, _FALLBACK[k]) for k in _PROPERTIES}

    # Derived soil features
    soil["awc"] = _estimate_awc(soil["clay"], soil["sand"])
    soil["indice_texture_sol"] = soil["clay"] / (soil["sand"] + 1e-6)
    soil["ratio_sand_clay"]    = soil["sand"] / (soil["clay"] + 1e-6)

    return soil


def _estimate_awc(clay_pct: float, sand_pct: float) -> float:
    """
    Saxton–Rawls approximation: available water capacity (% vol).
    AWC ≈ 0.299 − 0.251*sand − 0.195*clay (fraction → %)
    """
    s = clay_pct / 100
    c = sand_pct / 100
    awc = max(0.0, 0.299 - 0.251 * s - 0.195 * c) * 100
    return round(awc, 2)
