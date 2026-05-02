"""
Phase 3: Train classifier (extensif vs intensif) with spatial CV.
Strategy: find optimal NDWI threshold + XGBoost ensemble, pick best.
Run: python scripts/train_classifier.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES_CSV = DATA_DIR / "features.csv"
MODEL_PATH = MODEL_DIR / "classifier.json"
META_PATH = MODEL_DIR / "feature_names.json"

FEATURE_NAMES = [
    "ndvi_mean", "ndvi_std", "ndvi_p10", "ndvi_p90", "ndvi_amplitude",
    "ndwi_mean", "ndwi_std",
    "ndre_mean", "ndre_std",
    "area_ha",
    "ndvi_ndwi_ratio",
    "texture_proxy",
]
NDWI_IDX = FEATURE_NAMES.index("ndwi_mean")


def load_dataset():
    rows = list(csv.DictReader(FEATURES_CSV.open()))
    X, y, groups, splits = [], [], [], []
    for r in rows:
        X.append([float(r[f]) for f in FEATURE_NAMES])
        y.append(int(r["label"]))
        groups.append(int(r["cluster_id"]))
        splits.append(r["split"])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
        np.array(groups),
        splits,
    )


def print_confusion(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}")
    print("               pred extensif  pred intensif")
    print(f"  true extensif       {cm[0,0]:5d}          {cm[0,1]:5d}")
    print(f"  true intensif       {cm[1,0]:5d}          {cm[1,1]:5d}")


class NdwiThresholdClassifier:
    """Classify as intensif if ndwi_mean >= threshold, else extensif."""
    def __init__(self, threshold: float = -0.05):
        self.threshold = threshold

    def get_params(self, deep=True):
        return {"threshold": self.threshold}

    def fit(self, X, y):
        # Find optimal threshold via grid search on training data
        ndwi = X[:, NDWI_IDX]
        best_t, best_f1 = self.threshold, 0.0
        for t in np.linspace(ndwi.min() - 0.01, ndwi.max() + 0.01, 200):
            preds = (ndwi >= t).astype(int)
            f = f1_score(y, preds, average="macro", zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        self.threshold = best_t
        return self

    def predict(self, X):
        return (X[:, NDWI_IDX] >= self.threshold).astype(int)

    def predict_proba(self, X):
        ndwi = X[:, NDWI_IDX]
        # Soft probability based on distance from threshold
        dist = ndwi - self.threshold
        p_intensif = 1 / (1 + np.exp(-dist * 20))  # sigmoid scaled
        return np.column_stack([1 - p_intensif, p_intensif])


def spatial_cv_f1(make_model_fn, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    f1s = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        m = make_model_fn()
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        f1s.append(f1_score(y[val_idx], preds, average="macro", zero_division=0))
    return float(np.mean(f1s)), float(np.std(f1s))


def main():
    X, y, groups, splits = load_dataset()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"Dataset: {len(y)} samples — {n_neg} extensif, {n_pos} intensif")

    ndwi_vals = X[:, NDWI_IDX]
    print(f"NDWI: extensif mean={ndwi_vals[y==0].mean():.3f}  intensif mean={ndwi_vals[y==1].mean():.3f}")

    candidates = {
        "NDWI threshold":   lambda: NdwiThresholdClassifier(),
        "XGBoost depth=2":  lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.1,
            scale_pos_weight=n_neg / (n_pos + 1e-8),
            eval_metric="logloss", verbosity=0, random_state=42,
        ),
        "XGBoost depth=1":  lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=1, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / (n_pos + 1e-8),
            eval_metric="logloss", verbosity=0, random_state=42,
        ),
    }

    print("\n--- Spatial CV comparison ---")
    best_name, best_mean, best_fn = "", -1.0, None
    for name, fn in candidates.items():
        mean_f1, std_f1 = spatial_cv_f1(fn, X, y, groups)
        marker = ""
        if mean_f1 > best_mean:
            best_mean, best_name, best_fn = mean_f1, name, fn
            marker = " <-- best"
        print(f"  {name:<30} F1={mean_f1:.3f} ± {std_f1:.3f}{marker}")

    # Always also show per-fold detail for the NDWI threshold (most interpretable)
    print("\n--- NDWI threshold per-fold detail ---")
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        m = NdwiThresholdClassifier()
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
        ext_val = (y[val_idx] == 0).sum()
        int_val = (y[val_idx] == 1).sum()
        print(f"  Fold {fold+1}: threshold={m.threshold:.3f}  F1={f1:.3f}  (val: {ext_val} ext, {int_val} int)")

    print(f"\nSelected: {best_name}  (spatial CV F1={best_mean:.3f})")

    # Train final model on ALL data
    final_model = best_fn()  # type: ignore[misc]
    final_model.fit(X, y)

    # Save model
    if isinstance(final_model, NdwiThresholdClassifier):
        model_data = {
            "model_type": "ndwi_threshold",
            "threshold": float(final_model.threshold),
            "feature_names": FEATURE_NAMES,
            "classes": ["extensif", "intensif"],
        }
        MODEL_PATH.write_text(json.dumps(model_data, indent=2))
        print(f"\nNDWI threshold = {final_model.threshold:.4f}")
    else:
        final_model.save_model(MODEL_PATH)

    # Evaluate on test split
    test_mask = np.array([s == "test" for s in splits])
    if test_mask.sum() > 0:
        preds_test = final_model.predict(X[test_mask])
        print_confusion(y[test_mask], preds_test, "Test set confusion matrix:")
        print(f"\n{classification_report(y[test_mask], preds_test, target_names=['extensif','intensif'], zero_division=0)}")

    META_PATH.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "classes": ["extensif", "intensif"],
        "model_type": "ndwi_threshold" if isinstance(final_model, NdwiThresholdClassifier) else "xgboost",
        "threshold": float(final_model.threshold) if isinstance(final_model, NdwiThresholdClassifier) else None,
        "spatial_cv_f1": round(best_mean, 4),
        "best_model": best_name,
    }, indent=2))
    print(f"Metadata saved → {META_PATH}")
    print(f"\nSpatial CV F1 = {best_mean:.3f} {'✓ >= 0.70' if best_mean >= 0.70 else '✗ below 0.70'}")


if __name__ == "__main__":
    main()
