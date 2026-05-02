"""
Train one Prophet model per parcel using its real Sentinel-2 NDVI history.

Prophet learns two things from the historical time series:
  - Seasonal pattern : how NDVI rises and falls through the year
  - Long-term trend  : whether the parcel is slowly greening or drying over the years

This means the expected NDVI for July 2024 is NOT the same as July 2019 —
Prophet adjusts for the multi-year drift, which a simple monthly mean cannot do.

Input:  data/ndvi_history/<parcel_id>.csv   (from build_ndvi_history.py)
Output: models/prophet/<parcel_id>.json     (serialised Prophet model per parcel)
        models/prophet/meta.json            (parcel list + training stats)

Usage:
    venv/bin/python scripts/build_prophet_baselines.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# Suppress Prophet / Stan verbose output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "data" / "ndvi_history"
OUT_DIR = ROOT / "models" / "prophet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS = 24   # need at least 2 years of data to learn trend + seasonality


def train_parcel(csv_path: Path) -> dict:
    """
    Train a Prophet model on one parcel's NDVI history.

    Prophet expects a DataFrame with columns:
        ds  — datetime (date of observation)
        y   — value to forecast (NDVI)

    Returns a summary dict with training stats.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.rename(columns={"date": "ds", "ndvi_mean": "y"})
    df = df[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)

    if len(df) < MIN_OBS:
        raise ValueError(f"Only {len(df)} observations — need at least {MIN_OBS}")

    model = Prophet(
        yearly_seasonality=True,   # learns the annual NDVI cycle
        weekly_seasonality=False,  # no weekly pattern in olive NDVI
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # seasonal swings scale with the trend level
        changepoint_prior_scale=0.05,       # low = slow smooth trend, not abrupt jumps
        seasonality_prior_scale=10.0,       # allow flexible seasonal shape
        interval_width=0.95,                # 95% uncertainty interval
    )
    model.fit(df)

    # Compute in-sample residuals to get a realistic uncertainty estimate
    forecast = model.predict(df[["ds"]])
    residuals = df["y"].values - forecast["yhat"].values
    residual_std = float(residuals.std())

    # Serialise to JSON (Prophet's own format — no pickle needed)
    model_path = OUT_DIR / f"{csv_path.stem}.json"
    model_path.write_text(model_to_json(model))

    return {
        "n_observations": len(df),
        "date_start": str(df["ds"].min().date()),
        "date_end": str(df["ds"].max().date()),
        "ndvi_mean": round(float(df["y"].mean()), 4),
        "residual_std": round(residual_std, 4),
    }


def main() -> None:
    csv_files = sorted(HISTORY_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSVs found in {HISTORY_DIR}")
        print("Run build_ndvi_history.py first.")
        return

    print(f"Training Prophet models for {len(csv_files)} parcels ...\n")

    meta: dict[str, dict] = {}
    failed = 0

    for i, csv_path in enumerate(csv_files, 1):
        pid = csv_path.stem
        print(f"[{i:02d}/{len(csv_files)}] {pid} ... ", end="", flush=True)
        try:
            stats = train_parcel(csv_path)
            meta[pid] = stats
            print(
                f"ok  n={stats['n_observations']}  "
                f"mean={stats['ndvi_mean']}  "
                f"residual_std={stats['residual_std']}"
            )
        except Exception as exc:
            print(f"SKIP — {exc}")
            failed += 1

    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{len(meta)} models saved → {OUT_DIR}/")
    if failed:
        print(f"{failed} parcels skipped (insufficient data)")
    print("Next step: restart the API — Prophet models load automatically.")


if __name__ == "__main__":
    main()
