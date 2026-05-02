"""
Per-parcel Prophet baseline service.

Loads the serialised Prophet model for a parcel and predicts the expected
NDVI for any given date, accounting for:
  - The parcel's seasonal NDVI cycle (May peak, July dip, etc.)
  - The parcel's long-term trend (slow greening or drying over the years)

This is the true unsupervised baseline: no labels, no synthetic data —
only what the parcel's own 6-year satellite history tells us.

Falls back to the Ridge model if no Prophet model exists for the parcel.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

_PROPHET_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "prophet"
_META_PATH = _PROPHET_DIR / "meta.json"


@lru_cache(maxsize=1)
def _load_meta() -> dict:
    if _META_PATH.exists():
        return json.loads(_META_PATH.read_text())
    return {}


@lru_cache(maxsize=64)
def _load_model(parcel_id: str):
    """Load and cache a Prophet model. Returns None if not found."""
    from prophet.serialize import model_from_json  # type: ignore

    path = _PROPHET_DIR / f"{parcel_id}.json"
    if not path.exists():
        return None
    return model_from_json(path.read_text())


def has_prophet_model(parcel_id: str) -> bool:
    return (_PROPHET_DIR / f"{parcel_id}.json").exists()


def get_expected_series(
    parcel_id: str,
    end_date_str: str,
    n_steps: int = 5,
    step_days: int = 14,
) -> list[tuple[float, float]] | None:
    """
    Predict expected (mean, std) NDVI for n_steps bi-weekly steps ending at end_date_str.
    Ordered oldest → newest.

    The std comes from the model's 95% uncertainty interval converted to sigma:
        sigma ≈ (yhat_upper - yhat_lower) / (2 × 1.96)

    Returns None if no Prophet model exists for this parcel.
    """
    model = _load_model(parcel_id)
    if model is None:
        return None

    end = date.fromisoformat(end_date_str)
    # Shift future dates back 1 year (GEE archive constraint)
    if end > date.today():
        end = end.replace(year=end.year - 1)

    # Build the list of dates to predict
    dates = [
        end - timedelta(days=(n_steps - 1 - i) * step_days)
        for i in range(n_steps)
    ]

    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    forecast = model.predict(future)

    result = []
    for _, row in forecast.iterrows():
        mean = round(float(row["yhat"]), 4)
        # Convert 95% CI to std: CI_width / (2 * 1.96)
        std = round(float((row["yhat_upper"] - row["yhat_lower"]) / 3.92), 4)
        std = max(std, 0.01)  # floor to avoid division by zero
        result.append((mean, std))

    return result


def anomaly_zscore(
    ndvi_observe: list[float],
    ndvi_attendu: list[float],
    stds: list[float],
) -> float:
    """
    Mean Z-score across the 5-step window.

    Z = (expected − observed) / std   per step
    score = mean of positive Z-scores

    Unbounded positive float. 0 = healthy. > 2 = anomalous.
    """
    scores = [
        (exp - obs) / std
        for obs, exp, std in zip(ndvi_observe, ndvi_attendu, stds)
        if std > 0
    ]
    if not scores:
        return 0.0
    return round(max(0.0, sum(scores) / len(scores)), 2)
