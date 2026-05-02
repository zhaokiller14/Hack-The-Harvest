"""
Download Sentinel-2 GeoTIFF patches + binary masks for each EZZAYRA parcel.
Run: python scripts/download_tiles.py
Output: data/tiles/images/<id>_<split>.tif  data/tiles/masks/<id>_<split>.tif
"""
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TILES_DIR = DATA_DIR / "tiles"
IMAGES_DIR = TILES_DIR / "images"
MASKS_DIR = TILES_DIR / "masks"
PARCELS_PATH = DATA_DIR / "parcels_labeled.geojson"

# Tunisia bounding boxes with no known olive groves (for negative training samples)
NEGATIVE_BBOXES = [
    (8.0, 33.5, 8.5, 34.0),
    (9.0, 33.0, 9.5, 33.5),
    (8.5, 37.0, 9.0, 37.5),
    (10.0, 37.0, 10.5, 37.5),
    (7.5, 34.5, 8.0, 35.0),
    (9.5, 30.5, 10.0, 31.0),
    (10.5, 30.0, 11.0, 30.5),
]

PATCH_SIZE_PX = 256
# May-June: maximum olive/soil spectral contrast (jury spec)
DATE_START = "2025-05-01"
DATE_END = "2025-06-30"
CLOUD_PCT = 40  # pre-filter; SCL mask applied per-pixel after


def _scl_masked_composite(bbox: "ee.Geometry") -> "ee.Image":
    """
    Build a cloud-free median composite using the SCL band for pixel-level masking.
    SCL clear classes: 4=vegetation, 5=bare soil, 6=water, 7=unclassified,
                       11=snow (kept to avoid data gaps in sparse scenes).
    Cloud/shadow classes masked out: 1,2,3,8,9,10.
    """
    import ee

    def mask_scl(img):
        scl = img.select("SCL")
        clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
        return img.updateMask(clear).select(["B2", "B3", "B4", "B8", "B11"])

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT))
        .map(mask_scl)
        .median()
    )


def _fetch_and_resize(s2: "ee.Image", bbox: "ee.Geometry") -> "np.ndarray | None":
    """Download sampleRectangle and resize to PATCH_SIZE_PX. Returns [5,H,W] float32 or None."""
    import ee
    import cv2

    try:
        data = s2.sampleRectangle(region=bbox, defaultValue=0).getInfo()
    except Exception as e:
        return None

    bands = ["B2", "B3", "B4", "B8", "B11"]
    arrays = [np.array(data["properties"][b], dtype=np.float32) / 10000.0 for b in bands]
    img = np.clip(np.stack(arrays, axis=0), 0, 1)

    if img.shape[1] != PATCH_SIZE_PX or img.shape[2] != PATCH_SIZE_PX:
        img = np.stack([
            cv2.resize(img[i], (PATCH_SIZE_PX, PATCH_SIZE_PX), interpolation=cv2.INTER_LINEAR)
            for i in range(5)
        ], axis=0)
    return img


def download_patch(parcel_id: str, coords: list, label: int, split: str) -> bool:
    import ee
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
    from shapely.geometry import Polygon, mapping

    ee.Initialize(project="devflow-443322")

    shp = Polygon([[c[0], c[1]] for c in coords])
    minx, miny, maxx, maxy = shp.bounds

    pad_deg = 0.025
    minx -= pad_deg; miny -= pad_deg; maxx += pad_deg; maxy += pad_deg

    bbox = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    s2 = _scl_masked_composite(bbox)
    img = _fetch_and_resize(s2, bbox)
    if img is None:
        print(f"  GEE error for {parcel_id}")
        return False

    transform = from_bounds(minx, miny, maxx, maxy, PATCH_SIZE_PX, PATCH_SIZE_PX)
    img_path = IMAGES_DIR / f"{parcel_id}_{split}.tif"
    with rasterio.open(img_path, "w", driver="GTiff",
                       height=PATCH_SIZE_PX, width=PATCH_SIZE_PX,
                       count=5, dtype="float32", crs="EPSG:4326", transform=transform) as dst:
        dst.write(img)

    mask_arr = np.zeros((PATCH_SIZE_PX, PATCH_SIZE_PX), dtype=np.uint8)
    if label == 1:
        mask_arr = rasterize(
            [(mapping(shp), 1)],
            out_shape=(PATCH_SIZE_PX, PATCH_SIZE_PX),
            transform=transform, fill=0, dtype=np.uint8,
        )

    mask_path = MASKS_DIR / f"{parcel_id}_{split}.tif"
    with rasterio.open(mask_path, "w", driver="GTiff",
                       height=PATCH_SIZE_PX, width=PATCH_SIZE_PX,
                       count=1, dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
        dst.write(mask_arr[np.newaxis, :, :])

    return True


def download_negative(idx: int, bbox_tuple: tuple) -> bool:
    import ee
    import rasterio
    from rasterio.transform import from_bounds

    ee.Initialize(project="devflow-443322")
    minx, miny, maxx, maxy = bbox_tuple
    bbox = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    s2 = _scl_masked_composite(bbox)
    img = _fetch_and_resize(s2, bbox)
    if img is None:
        print(f"  GEE error for negative_{idx}")
        return False

    transform = from_bounds(minx, miny, maxx, maxy, PATCH_SIZE_PX, PATCH_SIZE_PX)
    img_path = IMAGES_DIR / f"negative_{idx}_train.tif"
    with rasterio.open(img_path, "w", driver="GTiff",
                       height=PATCH_SIZE_PX, width=PATCH_SIZE_PX,
                       count=5, dtype="float32", crs="EPSG:4326", transform=transform) as dst:
        dst.write(img)

    mask_arr = np.zeros((1, PATCH_SIZE_PX, PATCH_SIZE_PX), dtype=np.uint8)
    mask_path = MASKS_DIR / f"negative_{idx}_train.tif"
    with rasterio.open(mask_path, "w", driver="GTiff",
                       height=PATCH_SIZE_PX, width=PATCH_SIZE_PX,
                       count=1, dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
        dst.write(mask_arr)
    return True


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    geojson = json.loads(PARCELS_PATH.read_text())
    features = geojson["features"]
    print(f"Downloading {len(features)} positive patches...")

    ok, fail = 0, 0
    for feat in features:
        props = feat["properties"]
        parcel_id = props["id"]
        split = props["split"]
        label = props["label"]
        coords = feat["geometry"]["coordinates"][0]

        if list(IMAGES_DIR.glob(f"{parcel_id}_*.tif")):
            ok += 1
            continue

        print(f"  {parcel_id} ({props['systeme']}, {split})...", end=" ", flush=True)
        success = download_patch(parcel_id, coords, label, split)
        print("ok" if success else "FAIL")
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nPositive patches: {ok} ok, {fail} fail")
    print(f"Downloading {len(NEGATIVE_BBOXES)} negative patches...")

    ok_neg, fail_neg = 0, 0
    for i, bbox in enumerate(NEGATIVE_BBOXES):
        if list(IMAGES_DIR.glob(f"negative_{i}_*.tif")):
            ok_neg += 1
            continue
        print(f"  negative_{i}...", end=" ", flush=True)
        success = download_negative(i, bbox)
        print("ok" if success else "FAIL")
        if success:
            ok_neg += 1
        else:
            fail_neg += 1

    print(f"Negative patches: {ok_neg} ok, {fail_neg} fail")
    print(f"Total tiles in data/tiles/images/: {len(list(IMAGES_DIR.glob('*.tif')))}")


if __name__ == "__main__":
    main()
