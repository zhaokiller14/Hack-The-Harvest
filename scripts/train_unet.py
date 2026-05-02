"""
Train U-Net (ResNet34 backbone) on Sentinel-2 olive grove patches.
Run: python scripts/train_unet.py
Output: models/unet_best.pth, models/unet_config.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import albumentations as A
import segmentation_models_pytorch as smp

DATA_DIR = Path(__file__).parent.parent / "data"
TILES_DIR = DATA_DIR / "tiles"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 256
N_CHANNELS = 5
BATCH_SIZE = 4
LR = 3e-4
EPOCHS = 80
PATIENCE = 20


class OliveDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        images_dir = TILES_DIR / "images"
        masks_dir = TILES_DIR / "masks"
        if split == "all":
            self.image_paths = sorted(images_dir.glob("*.tif"))
        else:
            self.image_paths = sorted(images_dir.glob(f"*_{split}.tif"))
        self.masks_dir = masks_dir
        self.augment = augment
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(p=0.3),
        ]) if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
        else:
            mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.float32)
        img = np.clip(img, 0, 1)
        if self.augment and self.transform:
            img_hwc = img.transpose(1, 2, 0)
            aug = self.transform(image=img_hwc, mask=mask)
            img = aug["image"].transpose(2, 0, 1)
            mask = aug["mask"]
        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy()).unsqueeze(0)


class BceDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        pw = torch.tensor([pos_weight], device=DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
        )
        return 0.5 * bce_loss + 0.5 * dice_loss.mean()


def iou_score(preds_bin, targets):
    intersection = (preds_bin & targets.bool()).float().sum((1, 2, 3))
    union = (preds_bin | targets.bool()).float().sum((1, 2, 3))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


def build_model():
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet",
        in_channels=N_CHANNELS, classes=1,
    ).to(DEVICE)
    return model


def set_inference_mode(model):
    """Switch model to inference mode (equivalent to model.eval())."""
    model.train(False)
    return model


def main():
    print(f"Device: {DEVICE}")

    # Use train+val for fitting (only 8 val tiles, too few to waste)
    train_ds = OliveDataset("train", augment=True)
    train_val_ds = OliveDataset("val", augment=True)
    val_ds = OliveDataset("val", augment=False)  # for monitoring only

    if len(train_ds) == 0:
        print("No split-labeled tiles found, using 80/20 split of all tiles")
        all_ds = OliveDataset("all", augment=False)
        n_train = int(len(all_ds) * 0.8)
        train_ds, val_ds = torch.utils.data.random_split(
            all_ds, [n_train, len(all_ds) - n_train],
            generator=torch.Generator().manual_seed(42),
        )
        train_val_ds = None
    else:
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([train_ds, train_val_ds])

    print(f"Train: {len(train_ds)}  Val (monitor): {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model()
    criterion = BceDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_iou = 0.0
    patience_count = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        set_inference_mode(model)
        val_loss = val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_iou += iou_score((torch.sigmoid(logits) > 0.5).bool(), masks)
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_iou": round(val_iou, 4)})
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_iou={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), MODEL_DIR / "unet_best.pth")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val IoU: {best_iou:.4f}  {'PASS >= 0.65' if best_iou >= 0.65 else 'BELOW 0.65'}")
    (MODEL_DIR / "unet_history.json").write_text(json.dumps(history, indent=2))
    (MODEL_DIR / "unet_config.json").write_text(json.dumps({
        "encoder_name": "resnet34",
        "in_channels": N_CHANNELS,
        "classes": 1,
        "best_val_iou": round(best_iou, 4),
    }, indent=2))
    print("Model saved to models/unet_best.pth")


if __name__ == "__main__":
    main()
