"""U-Net inference: GEE tile download -> model inference -> vectorize mask to GeoJSON features."""
import asyncio
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_N_CHANNELS = 5
_PATCH_SIZE = 256


def load_unet_model() -> bool:
    global _model
    weights_path = _MODEL_DIR / "unet_best.pth"
    if not weights_path.exists():
        print("[segmentation] unet_best.pth not found — segmentation disabled")
        return False
    try:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=_N_CHANNELS, classes=1,
        ).to(_DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=_DEVICE))
        model.train(False)  # inference mode
        _model = model
        print(f"[segmentation] U-Net loaded on {_DEVICE}")
        return True
    except Exception as e:
        print(f"[segmentation] Failed to load U-Net: {e}")
        return False


def is_loaded() -> bool:
    return _model is not None


def _download_tile_sync(polygon_geojson: dict[str, Any]) -> "np.ndarray | None":
    try:
        import ee
        import cv2
        from app.config import get_settings

        cfg = get_settings()
        ee.Initialize(project=cfg.gee_project)

        coords = polygon_geojson["coordinates"][0]
        lats = [c[1] for c in coords]
        lngs = [c[0] for c in coords]
        pad = 0.05
        bbox = ee.Geometry.Rectangle([
            min(lngs) - pad, min(lats) - pad,
            max(lngs) + pad, max(lats) + pad,
        ])

        def mask_scl(img):
            scl = img.select("SCL")
            clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
            return img.updateMask(clear).select(["B2", "B3", "B4", "B8", "B11"])

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(bbox)
            .filterDate(cfg.sentinel_date_start, cfg.sentinel_date_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.sentinel_cloud_pct))
            .map(mask_scl)
            .median()
        )

        data = s2.sampleRectangle(region=bbox, defaultValue=0).getInfo()
        bands = ["B2", "B3", "B4", "B8", "B11"]
        arrays = [np.array(data["properties"][b], dtype=np.float32) / 10000.0 for b in bands]
        img = np.clip(np.stack(arrays, axis=0), 0, 1)

        if img.shape[1] != _PATCH_SIZE or img.shape[2] != _PATCH_SIZE:
            img = np.stack([
                cv2.resize(img[i], (_PATCH_SIZE, _PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
                for i in range(_N_CHANNELS)
            ], axis=0)
        return img
    except Exception as e:
        print(f"[segmentation] GEE download failed: {e}")
        return None


def _run_inference_sync(img: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(img).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        return (torch.sigmoid(_model(x)) > 0.5).cpu().numpy().squeeze().astype(np.uint8)


def _vectorize_mask_sync(mask: np.ndarray, polygon_geojson: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import cv2
        import shapely.geometry as sg
        from shapely.geometry import mapping

        coords = polygon_geojson["coordinates"][0]
        lats = [c[1] for c in coords]
        lngs = [c[0] for c in coords]
        pad = 0.05
        min_lng = min(lngs) - pad
        max_lng = max(lngs) + pad
        min_lat = min(lats) - pad
        max_lat = max(lats) + pad

        lng_scale = (max_lng - min_lng) / _PATCH_SIZE
        lat_scale = (max_lat - min_lat) / _PATCH_SIZE

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            pts = cnt.squeeze()
            if pts.ndim == 1:
                pts = pts[np.newaxis, :]
            ring = []
            for pt in pts:
                px, py = int(pt[0]), int(pt[1])
                ring.append([min_lng + px * lng_scale, max_lat - py * lat_scale])
            if len(ring) < 3:
                continue
            ring.append(ring[0])
            shp = sg.Polygon(ring)
            if not shp.is_valid:
                shp = shp.buffer(0)
            if shp.is_empty or shp.area < 1e-8:
                continue
            area_ha = shp.area * (111320 ** 2) * 0.85 / 10000
            features.append({
                "type": "Feature",
                "geometry": mapping(shp),
                "properties": {
                    "id": f"unet_{len(features)}",
                    "area_ha": round(area_ha, 2),
                    "source": "unet",
                },
            })
        return features
    except Exception as e:
        print(f"[segmentation] Vectorize error: {e}")
        return []


async def segment_polygon(polygon_geojson: dict[str, Any]) -> list[dict[str, Any]]:
    """Full pipeline: GEE download -> U-Net inference -> vectorize. Returns detected grove features."""
    if _model is None:
        return []
    loop = asyncio.get_event_loop()
    img = await loop.run_in_executor(None, partial(_download_tile_sync, polygon_geojson))
    if img is None:
        return []
    mask = await loop.run_in_executor(None, partial(_run_inference_sync, img))
    return await loop.run_in_executor(None, partial(_vectorize_mask_sync, mask, polygon_geojson))
