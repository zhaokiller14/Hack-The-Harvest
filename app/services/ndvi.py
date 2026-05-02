"""NDVI / NDWI / NDRE computation helpers (numpy-vectorised)."""
import numpy as np


def compute_ndvi(nir: list[float], red: list[float]) -> float:
    """Mean NDVI over a parcel. Expects reflectance values (0–1 or 0–10 000)."""
    nir_arr = np.asarray(nir, dtype=float)
    red_arr = np.asarray(red, dtype=float)
    ndvi = (nir_arr - red_arr) / (nir_arr + red_arr + 1e-8)
    return float(np.nanmean(ndvi))


def compute_ndwi(nir: list[float], swir: list[float]) -> float:
    """Mean NDWI (water index, B8/NIR vs B11/SWIR)."""
    nir_arr = np.asarray(nir, dtype=float)
    swir_arr = np.asarray(swir, dtype=float)
    ndwi = (nir_arr - swir_arr) / (nir_arr + swir_arr + 1e-8)
    return float(np.nanmean(ndwi))


def compute_ndre(nir: list[float], red_edge: list[float]) -> float:
    """Mean NDRE (red-edge chlorophyll index, B8/NIR vs B5/RedEdge)."""
    nir_arr = np.asarray(nir, dtype=float)
    re_arr = np.asarray(red_edge, dtype=float)
    ndre = (nir_arr - re_arr) / (nir_arr + re_arr + 1e-8)
    return float(np.nanmean(ndre))


def ndvi_parcel_stats(nir: list[float], red: list[float]) -> dict[str, float]:
    """Return mean, std, p10, p90 NDVI for a parcel (interior pixels only)."""
    nir_arr = np.asarray(nir, dtype=float)
    red_arr = np.asarray(red, dtype=float)
    ndvi = (nir_arr - red_arr) / (nir_arr + red_arr + 1e-8)
    valid = ndvi[np.isfinite(ndvi)]
    if len(valid) == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
    }
