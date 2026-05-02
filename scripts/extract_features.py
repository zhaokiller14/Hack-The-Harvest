"""
Phase 2: Extract 12 Sentinel-2 features per parcel via Google Earth Engine.
Run: python scripts/extract_features.py
Requires: earthengine-api, `earthengine authenticate` already done.
"""
import csv
import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
GEOJSON_PATH = DATA_DIR / "parcels_labeled.geojson"
FEATURES_CSV = DATA_DIR / "features.csv"
CACHE_JSON = DATA_DIR / "feature_cache.json"

DATE_START = "2024-05-01"
DATE_END = "2024-06-30"

FEATURE_NAMES = [
    "ndvi_mean", "ndvi_std", "ndvi_p10", "ndvi_p90", "ndvi_amplitude",
    "ndwi_mean", "ndwi_std",
    "ndre_mean", "ndre_std",
    "area_ha",
    "ndvi_ndwi_ratio",
    "texture_proxy",
]


def get_features_gee(polygon_geojson: dict, area_ha: float) -> dict[str, float]:
    """Call GEE to get spectral stats for a single parcel polygon."""
    import ee  # type: ignore

    aoi = ee.Geometry.Polygon(polygon_geojson["coordinates"])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
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

    def g(key: str, default: float = 0.0) -> float:
        return float(stats.get(key, default) or default)

    ndvi_mean = g("ndvi_mean")
    ndvi_std = g("ndvi_stdDev")
    ndvi_p10 = g("ndvi_p10")
    ndvi_p90 = g("ndvi_p90")
    ndwi_mean = g("ndwi_mean")
    ndwi_std = g("ndwi_stdDev")
    ndre_mean = g("ndre_mean")
    ndre_std = g("ndre_stdDev")

    return {
        "ndvi_mean": ndvi_mean,
        "ndvi_std": ndvi_std,
        "ndvi_p10": ndvi_p10,
        "ndvi_p90": ndvi_p90,
        "ndvi_amplitude": ndvi_p90 - ndvi_p10,
        "ndwi_mean": ndwi_mean,
        "ndwi_std": ndwi_std,
        "ndre_mean": ndre_mean,
        "ndre_std": ndre_std,
        "area_ha": area_ha,
        "ndvi_ndwi_ratio": ndvi_mean / (ndwi_mean + 1e-8),
        "texture_proxy": ndvi_std / (ndvi_mean + 1e-8),
    }


def main():
    import ee  # type: ignore
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else None
    ee.Initialize(project=project)

    geojson = json.loads(GEOJSON_PATH.read_text())
    features = geojson["features"]
    print(f"Extracting features for {len(features)} parcels...")

    rows: list[dict] = []
    cache: dict[str, dict] = {}

    for i, feat in enumerate(features):
        parcel_id = feat["properties"]["id"]
        systeme = feat["properties"]["systeme"]
        label = feat["properties"]["label"]
        area_ha = feat["properties"]["area_ha"]
        split = feat["properties"]["split"]
        cluster_id = feat["properties"]["cluster_id"]

        print(f"  [{i+1}/{len(features)}] {parcel_id} ({systeme}) ...", end=" ", flush=True)
        try:
            spectral = get_features_gee(feat["geometry"], area_ha)
            print(f"ndvi_mean={spectral['ndvi_mean']:.3f}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            spectral = {k: 0.0 for k in FEATURE_NAMES}

        row = {
            "id": parcel_id,
            "systeme": systeme,
            "label": label,
            "split": split,
            "cluster_id": cluster_id,
            **spectral,
        }
        rows.append(row)
        cache[parcel_id] = spectral

        # Small pause to avoid GEE rate limits
        time.sleep(0.5)

    # Write CSV
    fieldnames = ["id", "systeme", "label", "split", "cluster_id"] + FEATURE_NAMES
    with FEATURES_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved features → {FEATURES_CSV}")

    # Write cache (parcel_id → feature dict)
    CACHE_JSON.write_text(json.dumps(cache, indent=2))
    print(f"Saved cache → {CACHE_JSON}")


if __name__ == "__main__":
    main()
