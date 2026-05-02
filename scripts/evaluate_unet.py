"""
Evaluate trained U-Net on test split. Saves metrics and confusion matrix image.
Run: python scripts/evaluate_unet.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import rasterio
import segmentation_models_pytorch as smp

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
DOCS_DIR = Path(__file__).parent.parent / "docs"
TILES_DIR = DATA_DIR / "tiles"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 256
N_CHANNELS = 5


def build_model():
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=N_CHANNELS, classes=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "unet_best.pth", map_location=DEVICE))
    model.train(False)  # inference mode
    return model


def load_tiles(split: str):
    images_dir = TILES_DIR / "images"
    masks_dir = TILES_DIR / "masks"
    img_paths = sorted(images_dir.glob(f"*_{split}.tif"))
    if not img_paths:
        img_paths = [p for p in sorted(images_dir.glob("*.tif")) if "negative" not in p.stem]
    tiles = []
    for img_path in img_paths:
        mask_path = masks_dir / img_path.name
        with rasterio.open(img_path) as src:
            img = np.clip(src.read().astype(np.float32), 0, 1)
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
        tiles.append((img, mask))
    return tiles


def compute_metrics(model, tiles):
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for img, mask in tiles:
            x = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
            pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy().squeeze()
            gt = mask > 0.5
            tp += int(np.logical_and(pred, gt).sum())
            fp += int(np.logical_and(pred, ~gt).sum())
            fn += int(np.logical_and(~pred, gt).sum())
            tn += int(np.logical_and(~pred, ~gt).sum())

    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "iou": round(float(iou), 4), "precision": round(float(precision), 4),
        "recall": round(float(recall), 4), "f1": round(float(f1), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def save_confusion_image(metrics: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = metrics["confusion"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Fond", "Pred: Olivier"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Vrai: Fond", "Vrai: Olivier"])
        ax.set_title(f"U-Net — IoU={metrics['iou']:.3f}  F1={metrics['f1']:.3f}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=14)
        plt.tight_layout()
        out = DOCS_DIR / "unet_confusion_matrix.png"
        plt.savefig(out, dpi=120)
        print(f"Confusion matrix image saved to {out}")
    except ImportError:
        print("matplotlib not available, skipping image")


def main():
    model = build_model()
    tiles = load_tiles("test")
    if not tiles:
        print("No test tiles found, using all non-negative tiles")
        tiles = load_tiles("all")
    print(f"Evaluating on {len(tiles)} tiles...")
    metrics = compute_metrics(model, tiles)
    print(f"IoU={metrics['iou']:.4f}  precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    cm = metrics["confusion"]
    print(f"Confusion: TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")
    print(f"{'PASS IoU >= 0.65' if metrics['iou'] >= 0.65 else 'BELOW target IoU 0.65'}")

    out_path = MODEL_DIR / "unet_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {out_path}")
    save_confusion_image(metrics)


if __name__ == "__main__":
    main()
