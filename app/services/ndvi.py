"""NDVI / NDWI / NDRE computation helpers."""


def compute_ndvi(nir: list[float], red: list[float]) -> float:
    """Mean NDVI over a parcel. Replace with real raster math."""
    # TODO: numpy vectorised (nir - red) / (nir + red + 1e-8)
    return 0.62


def compute_ndwi(nir: list[float], swir: list[float]) -> float:
    return 0.18


def compute_ndre(nir: list[float], red_edge: list[float]) -> float:
    return 0.44
