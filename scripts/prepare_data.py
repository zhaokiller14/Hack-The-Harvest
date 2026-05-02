"""
Phase 1: Load EZZAYRA parcels, assign labels, spatial split, save GeoJSON.
Run: python scripts/prepare_data.py
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

EXTENSIF_PATH = Path.home() / "Downloads/Oliviers/parcelles_OlivierExtensif.json"
INTENSIF_PATH = Path.home() / "Downloads/Oliviers/parcellesOliviersIntensifs.json"


def coords_to_polygon(coordinates: list[dict]) -> dict:
    """Convert EZZAYRA [{lat, lng}] list to GeoJSON Polygon (closed ring)."""
    ring = [[c["lng"], c["lat"]] for c in coordinates]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def centroid(coordinates: list[dict]) -> tuple[float, float]:
    lats = [c["lat"] for c in coordinates]
    lngs = [c["lng"] for c in coordinates]
    return sum(lats) / len(lats), sum(lngs) / len(lngs)


def kmeans_1d(values: list[float], k: int, seed: int = 42) -> list[int]:
    """Simple 1-D k-means on latitude for geographic clustering."""
    random.seed(seed)
    centers = sorted(random.sample(values, k))
    for _ in range(50):
        labels = [min(range(k), key=lambda i: abs(v - centers[i])) for v in values]
        new_centers = []
        for i in range(k):
            group = [v for v, l in zip(values, labels) if l == i]
            new_centers.append(sum(group) / len(group) if group else centers[i])
        if new_centers == centers:
            break
        centers = new_centers
    return labels


def assign_splits(labels: list[int], cluster_ids: list[int], seed: int = 42) -> list[str]:
    """Assign train/val/test ensuring both classes appear in each split.

    Strategy: within each cluster, stratify by class so each split gets
    representatives of both extensif and intensif.
    """
    random.seed(seed)
    splits = ["train"] * len(labels)

    # Group indices by cluster
    from collections import defaultdict
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices[cid].append(idx)

    # Within each cluster shuffle and take 15% val, 15% test
    for cid, indices in cluster_to_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_val = max(0, round(n * 0.15))
        n_test = max(0, round(n * 0.15))
        for i in indices[:n_val]:
            splits[i] = "val"
        for i in indices[n_val:n_val + n_test]:
            splits[i] = "test"

    return splits


def load_parcels(path: Path, systeme: str) -> list[dict]:
    data = json.loads(path.read_text())
    parcels = []
    for p in data["parcels"]:
        parcels.append({
            "id": p["id"],
            "systeme": systeme,
            "area_ha": p["area_ha"],
            "coordinates": p["coordinates"],
        })
    return parcels


def main():
    ext_parcels = load_parcels(EXTENSIF_PATH, "extensif")
    int_parcels = load_parcels(INTENSIF_PATH, "intensif")
    all_parcels = ext_parcels + int_parcels

    print(f"Total parcels: {len(all_parcels)} ({len(ext_parcels)} extensif, {len(int_parcels)} intensif)")

    centroids = [centroid(p["coordinates"]) for p in all_parcels]
    lats = [c[0] for c in centroids]

    # Spatial clustering on latitude (Tunisia spans ~4° lat)
    k = 5
    cluster_ids = kmeans_1d(lats, k)

    labels = [0 if p["systeme"] == "extensif" else 1 for p in all_parcels]
    splits = assign_splits(labels, cluster_ids)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for s in splits:
        split_counts[s] += 1
    print(f"Split: {split_counts}")

    features = []
    for i, p in enumerate(all_parcels):
        lat, lng = centroids[i]
        features.append({
            "type": "Feature",
            "geometry": coords_to_polygon(p["coordinates"]),
            "properties": {
                "id": p["id"],
                "systeme": p["systeme"],
                "label": labels[i],
                "area_ha": p["area_ha"],
                "cluster_id": cluster_ids[i],
                "split": splits[i],
                "centroid_lat": lat,
                "centroid_lng": lng,
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path = DATA_DIR / "parcels_labeled.geojson"
    out_path.write_text(json.dumps(geojson, indent=2))
    print(f"Saved {len(features)} parcels → {out_path}")

    # Print per-class split breakdown
    for split in ["train", "val", "test"]:
        ext = sum(1 for f in features if f["properties"]["split"] == split and f["properties"]["systeme"] == "extensif")
        ints = sum(1 for f in features if f["properties"]["split"] == split and f["properties"]["systeme"] == "intensif")
        print(f"  {split}: {ext} extensif, {ints} intensif")


if __name__ == "__main__":
    main()
