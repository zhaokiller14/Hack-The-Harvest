"""
Train MCDropoutMLP on TomateProduction data and save mlp_model.pt
with real feature column names.

Pipeline mirrors notebooks/task-2/track2_rendement.ipynb:
  - Merge dev + hold (243 parcels)
  - Drop 14 near-constant features (>80% same value) → ~187 features
  - Log-transform target (best model per Section 10)
  - Stratified 5-fold CV (yield class as stratum)
  - 3× oversample faible class in training
  - Train MCDropoutMLP per fold, collect OOF MAPE
  - Retrain on full data, save checkpoint
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_percentage_error
from pathlib import Path

from app.services.mlp_model import MCDropoutMLP

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "TomateProduction"
OUT_PATH = _ROOT / "mlp_model.pt"

TARGET = "rendement_tha"
META   = ["polygon_id", "polygon_name", "kfold"]
DROP   = META + [TARGET]

# ── Load & merge ───────────────────────────────────────────────────────────────
dev  = pd.read_csv(DATA_DIR / "dev_set_07.csv")
hold = pd.read_csv(DATA_DIR / "hold_set_07.csv")
hold_aligned = hold.reindex(columns=dev.columns, fill_value=0)
full = pd.concat([dev, hold_aligned], ignore_index=True)
print(f"Merged: {full.shape[0]} rows × {full.shape[1]} cols")

# ── Feature selection ──────────────────────────────────────────────────────────
all_features = [c for c in dev.columns if c not in DROP]
X_full = full[all_features].copy()
y_full = full[TARGET].copy()

# Drop near-constant features (>80% same value)
threshold = 0.80
near_const = [
    col for col in all_features
    if X_full[col].value_counts(normalize=True).iloc[0] > threshold
]
print(f"Dropping {len(near_const)} near-constant features: {near_const}")
feature_cols = [c for c in all_features if c not in near_const]
X_full = X_full[feature_cols]
print(f"Features after cleaning: {len(feature_cols)}")

# ── Yield classes for stratification ──────────────────────────────────────────
low  = y_full.quantile(0.33)
high = y_full.quantile(0.67)

def yield_class(v):
    if v < low:    return 0   # faible
    elif v < high: return 1   # moyen
    else:          return 2   # fort

strata = y_full.apply(yield_class)
print(f"Yield class counts: {strata.value_counts().sort_index().to_dict()}")
print(f"Thresholds: faible < {low:.1f}, fort > {high:.1f}")

# ── Log-transform target (best model from Section 10) ─────────────────────────
y_log = np.log1p(y_full.values)

# ── StandardScaler fit on full data (saved in checkpoint) ─────────────────────
scaler_mean = X_full.values.astype(np.float32).mean(axis=0)
scaler_std  = X_full.values.astype(np.float32).std(axis=0)
scaler_std[scaler_std == 0] = 1.0   # avoid div-by-zero

X_scaled = (X_full.values.astype(np.float32) - scaler_mean) / scaler_std

# ── Training helpers ───────────────────────────────────────────────────────────
def train_one_fold(X_tr, y_tr, X_val, y_val, n_features, epochs=200, lr=1e-3, batch=32):
    model = MCDropoutMLP(n_features, hidden=[256, 128, 64], dropout_rate=0.3)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss()

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    best_val_loss = float("inf")
    best_state    = None
    patience, wait = 20, 0

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss_fn(pred, yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = loss_fn(val_pred, torch.tensor(y_val, dtype=torch.float32)).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ── 5-fold stratified CV ───────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(full))

print("\n=== Stratified CV (MLP, log-target) ===")
for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_scaled, strata)):
    X_tr_raw = X_scaled[train_idx]
    y_tr_raw = y_log[train_idx]
    X_val    = X_scaled[val_idx]
    y_val    = y_log[val_idx]

    # Oversample faible 3× in training fold
    tr_df = pd.DataFrame(X_tr_raw)
    tr_df["__y__"]   = y_tr_raw
    tr_df["__cls__"] = strata.values[train_idx]
    faible_rows = tr_df[tr_df["__cls__"] == 0]
    if len(faible_rows) > 0:
        upsampled = resample(faible_rows, replace=True, n_samples=len(faible_rows)*3, random_state=42)
        tr_df = pd.concat([tr_df, upsampled]).reset_index(drop=True)
    X_tr = tr_df.drop(columns=["__y__", "__cls__"]).values.astype(np.float32)
    y_tr = tr_df["__y__"].values.astype(np.float32)

    model = train_one_fold(X_tr, y_tr, X_val, y_val, n_features=len(feature_cols))

    with torch.no_grad():
        model.eval()
        preds_log = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    oof_preds[val_idx] = np.expm1(preds_log)

    # Per-fold MAPE
    y_val_real = y_full.values[val_idx]
    fold_mape  = mean_absolute_percentage_error(y_val_real, np.clip(oof_preds[val_idx], 1, None)) * 100
    print(f"  Fold {fold_id}: MAPE = {fold_mape:.1f}%")

# OOF MAPE by class
oof_df = pd.DataFrame({"true": y_full.values, "pred": np.clip(oof_preds, 1, None)})
oof_df["classe"] = oof_df["true"].apply(yield_class)
print("\n=== OOF MAPE by class ===")
for cls, name in [(0, "faible"), (1, "moyen"), (2, "fort")]:
    sub = oof_df[oof_df["classe"] == cls]
    m   = mean_absolute_percentage_error(sub["true"], sub["pred"]) * 100
    print(f"  {name}: {m:.1f}%  ({len(sub)} parcels)")
global_mape = mean_absolute_percentage_error(oof_df["true"], oof_df["pred"]) * 100
print(f"  global: {global_mape:.1f}%")

# ── Final model: retrain on full data ─────────────────────────────────────────
print("\nRetraining on full dataset...")

# Oversample faible 3×
full_df = pd.DataFrame(X_scaled)
full_df["__y__"]   = y_log
full_df["__cls__"] = strata.values
faible_rows = full_df[full_df["__cls__"] == 0]
if len(faible_rows) > 0:
    upsampled = resample(faible_rows, replace=True, n_samples=len(faible_rows)*3, random_state=42)
    full_df = pd.concat([full_df, upsampled]).reset_index(drop=True)

X_final = full_df.drop(columns=["__y__", "__cls__"]).values.astype(np.float32)
y_final = full_df["__y__"].values.astype(np.float32)

final_model = MCDropoutMLP(len(feature_cols), hidden=[256, 128, 64], dropout_rate=0.3)
opt    = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
loss_fn = nn.HuberLoss()
ds     = TensorDataset(torch.tensor(X_final), torch.tensor(y_final))
loader = DataLoader(ds, batch_size=32, shuffle=True)

final_model.train()
for epoch in range(300):
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(final_model(xb), yb).backward()
        opt.step()
    sched.step()
final_model.eval()

# ── Save checkpoint ────────────────────────────────────────────────────────────
torch.save({
    "model_state": final_model.state_dict(),
    "scaler_mean": scaler_mean,
    "scaler_std":  scaler_std,
    "feat_cols":   feature_cols,
}, OUT_PATH)
print(f"\nSaved → {OUT_PATH}  ({len(feature_cols)} features)")

# Quick sanity check
sample = torch.tensor(X_scaled[[0]], dtype=torch.float32)
mean_log, std_log = final_model.mc_predict(sample, n_passes=50)
pred_t = np.expm1(mean_log.item())
ci_lo  = np.expm1(mean_log.item() - 1.96 * std_log.item())
ci_hi  = np.expm1(mean_log.item() + 1.96 * std_log.item())
true_t = y_full.values[0]
print(f"\nSanity check — parcel 0:")
print(f"  True:      {true_t:.1f} t/ha")
print(f"  Predicted: {pred_t:.1f} t/ha  [{ci_lo:.1f}, {ci_hi:.1f}] 95% CI")
