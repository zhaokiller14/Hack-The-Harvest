"""
Fetch historical Sentinel-2 NDVI time series for all EZZAYRA parcels via GEE.

For each parcel, pulls every cloud-free Sentinel-2 scene from 2019-01-01 to
2024-12-31 and records the mean NDVI over the polygon interior.

Requirements:
    - ee.Authenticate() already done (run once per machine)
    - GEE_PROJECT set in .env

Output:
    data/ndvi_history/<parcel_id>.csv
    columns: date, doy, ndvi_mean

Runtime: ~15-30 min for all 49 parcels (one GEE call per parcel).

Usage:
    venv/bin/python scripts/build_ndvi_history.py
    venv/bin/python scripts/build_ndvi_history.py --resume   # skip already-done parcels
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Oliviers"
OUT_DIR = ROOT / "data" / "ndvi_history"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
CLOUD_THRESHOLD = 30  # % max cloud cover per scene


def load_parcels() -> list[dict]:
    records = []
    for fpath, systeme in [
        (DATA_DIR / "parcellesOliviersIntensifs.json", "intensif"),
        (DATA_DIR / "parcelles_OlivierExtensif.json", "extensif"),
    ]:
        data = json.loads(fpath.read_text())
        for p in data["parcels"]:
            # Convert {lat, lng} list → GeoJSON [lon, lat] ring
            ring = [[c["lng"], c["lat"]] for c in p["coordinates"]]
            ring.append(ring[0])  # close polygon
            records.append({
                "id": p["id"],
                "name": p["name"],
                "systeme": systeme,
                "area_ha": p.get("area_ha", 50.0),
                "geojson_coords": [ring],  # GeoJSON Polygon coordinates
            })
    return records


def fetch_parcel_series(parcel: dict, ee) -> pd.DataFrame:
    """
    Fetch all cloud-free Sentinel-2 observations for one parcel.
    Returns DataFrame with columns: date, doy, ndvi_mean.
    Each row = one scene (one satellite pass over the parcel).
    """
    aoi = ee.Geometry.Polygon(parcel["geojson_coords"])

    def extract_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
        mean_val = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=int(1e8),
            bestEffort=True,
        ).get("ndvi")
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "ndvi": mean_val,
        })

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
    )

    fc = col.map(extract_ndvi)
    features = fc.getInfo()["features"]

    rows = []
    for feat in features:
        props = feat["properties"]
        if props.get("ndvi") is None:
            continue
        d = props["date"]
        from datetime import date as _date
        doy = _date.fromisoformat(d).timetuple().tm_yday
        rows.append({
            "date": d,
            "doy": doy,
            "ndvi_mean": round(float(props["ndvi"]), 4),
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def main(resume: bool = False) -> None:
    import ee
    from dotenv import load_dotenv
    import os

    load_dotenv(ROOT / ".env")
    project = os.getenv("GEE_PROJECT", "")

    print("Initialising GEE …")
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    print("GEE ready.\n")

    parcels = load_parcels()
    print(f"{len(parcels)} parcels to process\n")

    for i, parcel in enumerate(parcels, 1):
        pid = parcel["id"]
        out_path = OUT_DIR / f"{pid}.csv"

        if resume and out_path.exists():
            print(f"[{i:02d}/{len(parcels)}] {parcel['name']} ({parcel['systeme']}) — skip (exists)")
            continue

        print(f"[{i:02d}/{len(parcels)}] {parcel['name']} ({parcel['systeme']}) … ", end="", flush=True)
        t0 = time.time()
        try:
            df = fetch_parcel_series(parcel, ee)
            df.to_csv(out_path, index=False)
            print(f"{len(df)} observations  ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"\nDone. CSVs saved to {OUT_DIR}/")
    print("Next step: venv/bin/python scripts/build_parcel_baselines.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip parcels already downloaded")
    args = parser.parse_args()
    main(resume=args.resume)
