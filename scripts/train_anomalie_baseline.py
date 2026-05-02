"""
Train Ridge regression NDVI baseline models for olive anomaly detection.

Generates synthetic but phenologically realistic training data for
Tunisia olive groves (2018–2024), then fits one Ridge model per system
(extensif / intensif / hyper_intensif).

Saved artefacts in models/:
    ridge_<system>.pkl        — sklearn Pipeline (StandardScaler + Ridge)
    residual_std_<system>.csv — per-parcel NDVI residual std for Z-score thresholds
    baseline_meta.json        — feature names, system stats, training date

Usage:
    python scripts/train_anomalie_baseline.py
"""
from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Oliviers"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Phenology parameters (healthy olive baseline) ──────────────────────────
# Peak NDVI in early May (DOY ~120–130).  Olives are evergreen → base > 0.
_PHENO: dict[str, dict] = {
    "extensif": {
        "base": 0.36, "amplitude": 0.09, "peak_doy": 115,
        # Weather sensitivity (rainfed → strongly rain-dependent)
        "rain_coef": 0.00080,   # +NDVI per mm rainfall in window
        "et0_coef": -0.00025,   # −NDVI per mm ET0
        "heat_coef": -0.00800,  # −NDVI per heat-stress day
        "noise_std": 0.018,
    },
    "intensif": {
        "base": 0.50, "amplitude": 0.07, "peak_doy": 125,
        # Irrigated → weakly rain-dependent
        "rain_coef": 0.00025,
        "et0_coef": -0.00008,
        "heat_coef": -0.00600,
        "noise_std": 0.014,
    },
    "hyper_intensif": {
        "base": 0.60, "amplitude": 0.05, "peak_doy": 130,
        # High-input → almost weather-independent
        "rain_coef": 0.00015,
        "et0_coef": -0.00004,
        "heat_coef": -0.00400,
        "noise_std": 0.011,
    },
}

# ── Tunisia seasonal weather normals (21-day window centred on each month) ─
# Indices 0–11 = Jan–Dec
#                      J    F    M    A    M    J    J    A    S    O    N    D
_RAIN_NORMAL_21D = [40., 35., 26., 17., 10.,  2.,  1.,  2., 17., 32., 42., 40.]   # mm
_ET0_NORMAL_21D  = [41., 44., 55., 69., 90., 107., 118., 114., 90., 65., 44., 37.] # mm
_TMAX_NORMAL     = [15., 17., 20., 24., 29., 34.,  37.,  37.,  32., 27., 21., 16.] # °C
_TMIN_NORMAL     = [ 7.,  8., 10., 13., 17., 21.,  24.,  24.,  20., 16., 11.,  8.] # °C

FEATURE_COLS = [
    "doy_sin", "doy_cos",
    "rainfall_21d", "et0_21d", "gdd_21d", "heat_stress_days",
    "area_ha_log",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _seasonal_ndvi(doy: int, systeme: str) -> float:
    p = _PHENO[systeme]
    phase = 2.0 * math.pi * (doy - p["peak_doy"]) / 365.0
    return p["base"] + p["amplitude"] * math.cos(phase)


def _synthetic_weather(center_date: date, rng: random.Random) -> dict:
    """Sample plausible 21-day weather stats for `center_date` using monthly normals + noise."""
    m = center_date.month - 1  # 0-indexed
    rain = max(0.0, rng.gauss(_RAIN_NORMAL_21D[m], _RAIN_NORMAL_21D[m] * 0.5))
    et0 = max(5.0, rng.gauss(_ET0_NORMAL_21D[m], _ET0_NORMAL_21D[m] * 0.15))
    tmax = rng.gauss(_TMAX_NORMAL[m], 2.5)
    tmin = rng.gauss(_TMIN_NORMAL[m], 2.0)
    gdd = max(0.0, (tmax + tmin) / 2.0 - 10.0) * 21
    heat = max(0, int(rng.gauss((tmax - 35) * 1.5, 1.0))) if tmax > 35 else 0
    return {"rainfall_mm": round(rain, 1), "et0_mm": round(et0, 1),
            "gdd": round(gdd, 1), "heat_stress_days": heat}


def _healthy_ndvi(doy: int, systeme: str, weather: dict, rng: random.Random) -> float:
    p = _PHENO[systeme]
    base = _seasonal_ndvi(doy, systeme)
    weather_effect = (
        p["rain_coef"] * weather["rainfall_mm"]
        + p["et0_coef"] * weather["et0_mm"]
        + p["heat_coef"] * weather["heat_stress_days"]
    )
    noise = rng.gauss(0.0, p["noise_std"])
    return max(0.05, min(0.95, base + weather_effect + noise))


def _build_features(doy: int, weather: dict, area_ha: float) -> dict:
    angle = 2.0 * math.pi * doy / 365.0
    return {
        "doy_sin": math.sin(angle),
        "doy_cos": math.cos(angle),
        "rainfall_21d": weather["rainfall_mm"],
        "et0_21d": weather["et0_mm"],
        "gdd_21d": weather["gdd"],
        "heat_stress_days": float(weather["heat_stress_days"]),
        "area_ha_log": math.log1p(area_ha),
    }


# ── Load parcel metadata ───────────────────────────────────────────────────

def load_parcels() -> list[dict]:
    records = []
    for fpath, systeme in [
        (DATA_DIR / "parcellesOliviersIntensifs.json", "intensif"),
        (DATA_DIR / "parcelles_OlivierExtensif.json", "extensif"),
    ]:
        data = json.loads(fpath.read_text())
        for p in data["parcels"]:
            records.append({
                "id": p["id"],
                "systeme": systeme,
                "area_ha": p.get("area_ha", 50.0),
            })
    # Add synthetic hyper_intensif parcels (none in raw data — extrapolate)
    for i in range(8):
        records.append({
            "id": f"synth_hi_{i}",
            "systeme": "hyper_intensif",
            "area_ha": float(random.randint(15, 60)),
        })
    return records


# ── Generate training dataset ──────────────────────────────────────────────

def generate_dataset(parcels: list[dict], seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    # 6 years × 26 bi-weekly observations per year
    start = date(2018, 1, 7)
    dates = [start + timedelta(days=14 * i) for i in range(6 * 26)]

    for parcel in parcels:
        systeme = parcel["systeme"]
        area_ha = parcel["area_ha"]
        for obs_date in dates:
            doy = obs_date.timetuple().tm_yday
            weather = _synthetic_weather(obs_date, rng)
            ndvi = _healthy_ndvi(doy, systeme, weather, rng)
            feats = _build_features(doy, weather, area_ha)
            rows.append({
                "parcel_id": parcel["id"],
                "systeme": systeme,
                "date": obs_date.isoformat(),
                "ndvi": round(ndvi, 4),
                **feats,
            })

    return pd.DataFrame(rows)


# ── Train per-system Ridge model ───────────────────────────────────────────

def train_system(df: pd.DataFrame, systeme: str) -> tuple[Pipeline, pd.Series, dict]:
    subset = df[df["systeme"] == systeme].copy()
    X = subset[FEATURE_COLS].values
    y = subset["ndvi"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"  [{systeme}]  MAE train={mae_train:.4f}  MAE test={mae_test:.4f}  R²={r2_test:.4f}")

    # Per-parcel residual std on full dataset (for Z-score thresholds)
    subset["ndvi_pred"] = pipe.predict(X)
    subset["residual"] = subset["ndvi"] - subset["ndvi_pred"]
    residual_std = subset.groupby("parcel_id")["residual"].std().fillna(
        subset["residual"].std()
    )

    stats = {
        "system_residual_std": float(subset["residual"].std()),
        "system_ndvi_mean": float(y.mean()),
        "system_ndvi_std": float(y.std()),
        "mae_test": float(mae_test),
        "r2_test": float(r2_test),
        "n_samples": int(len(subset)),
    }

    return pipe, residual_std, stats


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading parcel metadata …")
    parcels = load_parcels()
    print(f"  {len(parcels)} parcels loaded")

    print("Generating synthetic training dataset …")
    df = generate_dataset(parcels)
    print(f"  {len(df):,} samples generated")

    meta: dict = {"feature_cols": FEATURE_COLS, "systems": {}}

    for systeme in ["extensif", "intensif", "hyper_intensif"]:
        print(f"\nTraining {systeme} …")
        pipe, residual_std, stats = train_system(df, systeme)

        model_path = MODELS_DIR / f"ridge_{systeme}.pkl"
        joblib.dump(pipe, model_path)
        print(f"  Saved {model_path.name}")

        std_path = MODELS_DIR / f"residual_std_{systeme}.csv"
        residual_std.to_csv(std_path, header=True)
        print(f"  Saved {std_path.name}")

        meta["systems"][systeme] = stats

    meta_path = MODELS_DIR / "baseline_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {meta_path.name}")
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    main()
