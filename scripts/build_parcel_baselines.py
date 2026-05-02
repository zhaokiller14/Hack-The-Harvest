"""
Build per-parcel DOY (day-of-year) NDVI baselines from the fetched history CSVs.

For each parcel and each DOY 1-365, computes the historical mean and std of
NDVI using all observations within a ±30-day Gaussian-weighted window.
This gives a smooth seasonal curve per parcel — the true unsupervised baseline.

Input:  data/ndvi_history/<parcel_id>.csv   (from build_ndvi_history.py)
Output: data/parcel_baselines.json

Usage:
    venv/bin/python scripts/build_parcel_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "data" / "ndvi_history"
OUT_PATH = ROOT / "data" / "parcel_baselines.json"
PARCEL_JSON_DIR = ROOT / "data" / "Oliviers"

# Gaussian kernel half-width in days — smooths across ±30 days of DOY history
KERNEL_SIGMA = 15.0
KERNEL_WINDOW = 30      # only use observations within ±30 days
MIN_OBS = 5             # minimum observations needed to build a reliable baseline


def circular_doy_diff(doy_a: np.ndarray, doy_b: int) -> np.ndarray:
    """Shortest angular distance on a 365-day circle."""
    diff = np.abs(doy_a - doy_b)
    return np.minimum(diff, 365 - diff)


def build_parcel_baseline(df: pd.DataFrame) -> dict:
    """
    Given a parcel's full NDVI history, compute per-DOY statistics.

    Returns a dict with keys "1" … "365", each containing:
        mean  — expected healthy NDVI for that day of year
        std   — historical variability (used as Z-score denominator)
        n     — number of observations contributing
    """
    doys = df["doy"].values.astype(float)
    ndvi = df["ndvi_mean"].values.astype(float)

    baseline: dict[str, dict] = {}

    for target_doy in range(1, 366):
        diffs = circular_doy_diff(doys, target_doy)
        mask = diffs <= KERNEL_WINDOW

        if mask.sum() < MIN_OBS:
            # Not enough data for this DOY — mark as missing, filled later
            baseline[str(target_doy)] = {"mean": None, "std": None, "n": int(mask.sum())}
            continue

        weights = np.exp(-0.5 * (diffs[mask] / KERNEL_SIGMA) ** 2)
        w_sum = weights.sum()
        w_mean = float(np.sum(weights * ndvi[mask]) / w_sum)

        # Weighted standard deviation
        w_var = float(np.sum(weights * (ndvi[mask] - w_mean) ** 2) / w_sum)
        w_std = float(np.sqrt(w_var))

        baseline[str(target_doy)] = {
            "mean": round(w_mean, 4),
            "std": round(max(w_std, 0.01), 4),  # floor at 0.01 to avoid division by zero
            "n": int(mask.sum()),
        }

    # Fill missing DOYs by linear interpolation from neighbours
    _fill_missing(baseline)

    return baseline


def _fill_missing(baseline: dict) -> None:
    """Fill DOYs with None mean by interpolating from nearest valid neighbours."""
    means = [baseline[str(d)]["mean"] for d in range(1, 366)]
    stds = [baseline[str(d)]["std"] for d in range(1, 366)]

    # Find valid indices
    valid_idx = [i for i, v in enumerate(means) if v is not None]
    if not valid_idx:
        return

    all_idx = list(range(365))
    valid_means = [means[i] for i in valid_idx]
    valid_stds = [stds[i] for i in valid_idx]

    interp_means = np.interp(all_idx, valid_idx, valid_means)
    interp_stds = np.interp(all_idx, valid_idx, valid_stds)

    for i in range(365):
        if means[i] is None:
            baseline[str(i + 1)] = {
                "mean": round(float(interp_means[i]), 4),
                "std": round(float(interp_stds[i]), 4),
                "n": 0,
            }


def load_parcel_meta() -> dict[str, dict]:
    """Return {parcel_id: {systeme, area_ha}} for all parcels."""
    meta = {}
    for fpath, systeme in [
        (PARCEL_JSON_DIR / "parcellesOliviersIntensifs.json", "intensif"),
        (PARCEL_JSON_DIR / "parcelles_OlivierExtensif.json", "extensif"),
    ]:
        data = json.loads(fpath.read_text())
        for p in data["parcels"]:
            meta[p["id"]] = {"systeme": systeme, "area_ha": p.get("area_ha", 50.0)}
    return meta


def main() -> None:
    csv_files = sorted(HISTORY_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSVs found in {HISTORY_DIR}")
        print("Run build_ndvi_history.py first.")
        return

    print(f"Found {len(csv_files)} parcel history files")
    parcel_meta = load_parcel_meta()

    result: dict = {}
    skipped = 0

    for csv_path in csv_files:
        pid = csv_path.stem
        df = pd.read_csv(csv_path)

        if len(df) < MIN_OBS:
            print(f"  {pid}: only {len(df)} obs — skipped (need ≥{MIN_OBS})")
            skipped += 1
            continue

        baseline = build_parcel_baseline(df)
        meta = parcel_meta.get(pid, {})

        # Summary stats for the record
        valid = [v for v in baseline.values() if v["mean"] is not None]
        ndvi_values = [v["mean"] for v in valid]

        result[pid] = {
            "systeme": meta.get("systeme", "unknown"),
            "area_ha": meta.get("area_ha", 0.0),
            "n_observations": len(df),
            "ndvi_annual_mean": round(float(np.mean(ndvi_values)), 4),
            "ndvi_annual_std": round(float(np.std(ndvi_values)), 4),
            "doy_baseline": baseline,
        }
        print(f"  {pid}: {len(df)} obs  annual_mean={result[pid]['ndvi_annual_mean']:.3f}")

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n{len(result)} baselines saved → {OUT_PATH}")
    if skipped:
        print(f"{skipped} parcels skipped (insufficient data)")


if __name__ == "__main__":
    main()
