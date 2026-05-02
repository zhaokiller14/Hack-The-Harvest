"""Load trained model and predict olive grove system type."""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "classifier.json"
_META_PATH = _MODEL_DIR / "feature_names.json"

_model_type: str = "heuristic"
_feature_names: list[str] = []
_classes: list[str] = []
_ndwi_threshold: float = -0.054  # sensible default from data
_xgb_model: xgb.XGBClassifier | None = None
_ndwi_idx: int = 5  # index of ndwi_mean in FEATURE_NAMES


def load_model() -> None:
    global _model_type, _feature_names, _classes, _ndwi_threshold, _xgb_model, _ndwi_idx

    if not _META_PATH.exists():
        return

    meta = json.loads(_META_PATH.read_text())
    _feature_names = meta.get("feature_names", [])
    _classes = meta.get("classes", ["extensif", "intensif"])
    _model_type = meta.get("model_type", "heuristic")

    if "ndwi_mean" in _feature_names:
        _ndwi_idx = _feature_names.index("ndwi_mean")

    if _model_type == "ndwi_threshold":
        _ndwi_threshold = float(meta.get("threshold") or -0.054)

    elif _model_type == "xgboost":
        xgb_filename = meta.get("xgb_path", "classifier_xgb.json")
        xgb_path = _MODEL_DIR / xgb_filename
        if xgb_path.exists():
            try:
                _xgb_model = xgb.XGBClassifier()
                _xgb_model.load_model(xgb_path)
            except Exception as e:
                print(f"[classifier] Failed to load XGBoost from {xgb_path}: {e}")
        elif _MODEL_PATH.exists():
            try:
                content = json.loads(_MODEL_PATH.read_text())
                if "threshold" in content:
                    _model_type = "ndwi_threshold"
                    _ndwi_threshold = float(content.get("threshold", _ndwi_threshold))
                else:
                    _xgb_model = xgb.XGBClassifier()
                    _xgb_model.load_model(_MODEL_PATH)
            except Exception:
                pass

    elif _model_type == "sklearn":
        import joblib

        sk_path = _MODEL_DIR / "classifier.pkl"
        if sk_path.exists():
            _xgb_model = joblib.load(sk_path)

    print(
        f"[classifier] model_type={_model_type}"
        + (
            f"  threshold={_ndwi_threshold:.4f}"
            if _model_type == "ndwi_threshold"
            else ""
        )
    )


def _feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([[features.get(f, 0.0) for f in _feature_names]], dtype=np.float32)


def predict(features: dict[str, float]) -> tuple[str, float]:
    """Return (systeme, confidence)."""
    if _model_type == "ndwi_threshold":
        ndwi = features.get("ndwi_mean", 0.0)
        dist = ndwi - _ndwi_threshold
        p_intensif = float(1 / (1 + np.exp(-dist * 20)))
        if ndwi >= _ndwi_threshold:
            return "intensif", max(0.6, p_intensif)
        else:
            return "extensif", max(0.6, 1 - p_intensif)

    if _xgb_model is not None and _feature_names:
        X = _feature_vector(features)
        proba = _xgb_model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        return _classes[idx], float(proba[idx])

    return _heuristic(features)


def _heuristic(features: dict[str, float]) -> tuple[str, float]:
    ndwi = features.get("ndwi_mean", 0.0)
    if ndwi >= _ndwi_threshold:
        return "intensif", 0.65
    return "extensif", 0.65


def is_loaded() -> bool:
    return _model_type != "heuristic"
