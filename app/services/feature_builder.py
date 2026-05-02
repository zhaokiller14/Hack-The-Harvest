"""
Converts raw API outputs (weather, soil, sentinel bands, area) into the 187
z-scored feature columns expected by mlp_model.pt.

Reference normalization parameters (REF_MEAN, REF_STD) are estimated from
typical Tunisian tomato growing conditions and match the original dataset
distribution. They convert raw physical values → z-scores that are in the
same scale as the training CSV.
"""

from __future__ import annotations

import math
from typing import Any

SEASONS = ("s1", "s2", "s3", "s4")

# ── Reference distributions for raw physical values ───────────────────────────
# (mean, std) per season [s1, s2, s3, s4] for weather
_W_REF: dict[str, list[tuple[float, float]]] = {
    "gdd":               [(120, 50),  (260, 70),  (400, 80),  (520, 90)],
    "temp_mean":         [(14,  4),   (19,  4),   (24,  4),   (28,  4)],
    "temp_max":          [(19,  4),   (24,  4),   (30,  5),   (35,  5)],
    "temp_min":          [(9,   3),   (14,  3),   (18,  4),   (21,  4)],
    "temp_amplitude":    [(10,  3),   (10,  3),   (12,  3),   (14,  3)],
    "precip_cum":        [(42, 35),   (30, 28),   (15, 20),   (8,  15)],
    "et0_mean":          [(3.2, 1.0), (4.8, 1.2), (6.5, 1.5), (8.0, 1.5)],
    "solar_cum":         [(155, 40),  (210, 40),  (270, 45),  (310, 45)],
    "wind_speed_max":    [(28, 10),   (28, 10),   (30, 10),   (32, 10)],
    "humidity_mean":     [(75, 12),   (62, 12),   (50, 12),   (42, 14)],
    "humidity_max":      [(90,  8),   (82, 10),   (72, 12),   (65, 12)],
    "humidity_min":      [(55, 12),   (42, 12),   (28, 12),   (20, 12)],
    "freq_secheresse":   [(5,   6),   (10,  8),   (22,  8),   (27,  5)],
    "intensite_chaleur": [(0,   2),   (1,   3),   (6,   6),   (14,  8)],
    "stress_hydrique":   [(54, 40),   (114, 55),  (180, 65),  (232, 65)],
    "intensite_stress":  [(1.8, 1.5), (3.8, 2.5), (6.0, 3.5), (7.7, 3.5)],
    "precipitation_hours": [(8, 7),   (5,  6),    (3,  5),    (1,  3)],
    "soil_moisture":     [(0.78, 0.30), (0.47, 0.25), (0.23, 0.20), (0.10, 0.15)],
    "soil_temp":         [(12,  4),   (17,  4),   (22,  4),   (26,  4)],
}

# (mean, std) for vegetation indices per season
_V_REF: dict[str, list[tuple[float, float]]] = {
    "ndvi_mean": [(0.35, 0.12), (0.55, 0.15), (0.65, 0.15), (0.50, 0.15)],
    "ndvi_max":  [(0.50, 0.12), (0.70, 0.15), (0.78, 0.14), (0.65, 0.15)],
    "ndvi_std":  [(0.08, 0.05), (0.09, 0.05), (0.10, 0.05), (0.10, 0.05)],
    "evi_mean":  [(0.20, 0.08), (0.35, 0.10), (0.42, 0.12), (0.32, 0.12)],
    "evi_max":   [(0.30, 0.10), (0.48, 0.12), (0.55, 0.12), (0.45, 0.12)],
    "evi2_mean": [(0.22, 0.08), (0.38, 0.10), (0.45, 0.12), (0.35, 0.12)],
    "evi2_max":  [(0.32, 0.10), (0.52, 0.12), (0.58, 0.12), (0.48, 0.12)],
    "dswi_mean": [(0.50, 0.12), (0.60, 0.12), (0.65, 0.12), (0.55, 0.12)],
    "dswi_max":  [(0.65, 0.12), (0.75, 0.12), (0.80, 0.12), (0.70, 0.12)],
    "ndwi_mean": [(-0.20, 0.15), (-0.10, 0.15), (-0.05, 0.15), (-0.15, 0.15)],
    "ndwi_max":  [(-0.05, 0.15), (0.05, 0.15), (0.10, 0.15), (0.00, 0.15)],
    "nri_mean":  [(0.05, 0.03), (0.08, 0.04), (0.10, 0.04), (0.07, 0.04)],
    "nri_max":   [(0.10, 0.05), (0.15, 0.05), (0.18, 0.05), (0.12, 0.05)],
    "cloud_cover": [(0.10, 0.12), (0.08, 0.10), (0.05, 0.08), (0.05, 0.08)],
    "cloud_cover_sat": [(0.08, 0.10), (0.06, 0.08), (0.04, 0.07), (0.04, 0.07)],
}

# (mean, std) for static soil features (raw physical values)
_S_REF: dict[str, tuple[float, float]] = {
    "bdod":              (1.40,  0.18),
    "cec":               (18.0,  7.0),
    "cfvo":              (5.0,   5.0),
    "clay":              (28.0,  10.0),
    "nitrogen":          (0.90,  0.40),
    "phh2o":             (7.6,   0.55),
    "sand":              (42.0,  12.0),
    "silt":              (30.0,  10.0),
    "soc":               (10.0,  6.0),
    "awc":               (12.0,  4.0),
    "indice_texture_sol":(0.80,  0.50),
    "ratio_sand_clay":   (1.70,  0.80),
}

# area_polygon: raw hectares (NOT z-scored in dataset, model sees raw value)
_AREA_REF = None   # no z-scoring needed — model scaler handles it directly


def _zscore(raw: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (raw - mean) / std


def _season_zscore(raw: float, feature: str, season_idx: int,
                   ref: dict[str, list]) -> float:
    if feature not in ref:
        return 0.0
    mean, std = ref[feature][season_idx]
    return _zscore(raw, mean, std)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── Public entry point ────────────────────────────────────────────────────────

def build_features(
    weather: dict[str, Any],
    soil:    dict[str, float],
    bands:   dict[str, Any],
    area_ha: float,
) -> dict[str, float]:
    """
    Build the full 187-feature dict from raw API outputs.

    Parameters
    ----------
    weather : nested dict from fetch_weather()  { "s1": {...}, "s2": {...}, … }
    soil    : dict from fetch_soil()             { "bdod": 1.4, "clay": 28, … }
    bands   : dict from fetch_sentinel2()        { "ndvi_series": […], … }
    area_ha : parcel area in hectares

    Returns
    -------
    dict mapping feature_name → z-scored float ready for the model
    """
    feat: dict[str, float] = {}

    # ── 1. Area (raw, model scaler normalises it) ─────────────────────────────
    feat["area_polygon"] = area_ha

    # ── 2. Soil static features ───────────────────────────────────────────────
    for soil_key in _S_REF:
        if soil_key in soil:
            mean, std = _S_REF[soil_key]
            feat[soil_key] = _zscore(soil[soil_key], mean, std)

    # ── 3. Weather + vegetation per season ───────────────────────────────────
    ndvi_all: list[float] = []

    for si, skey in enumerate(SEASONS):
        w = weather.get(skey, {})

        # --- weather features
        for wf in _W_REF:
            col = f"{wf}_{skey}"
            raw = w.get(wf)
            feat[col] = _season_zscore(raw if raw is not None else _W_REF[wf][si][0],
                                       wf, si, _W_REF)

        # --- vegetation indices from Sentinel
        veg = _extract_veg(bands, si)
        for vf, raw in veg.items():
            col = f"{vf}_{skey}"
            feat[col] = _season_zscore(raw, vf, si, _V_REF)

        # z_ndvi: how far NDVI mean is from historical mean (already z-scored by _season_zscore)
        feat[f"z_ndvi_{skey}"] = feat.get(f"ndvi_mean_{skey}", 0.0)

        # ratio_ndvi: ndvi_max / ndvi_mean (raw ratio, z-score relative to 1.3 ± 0.2)
        ndvi_mean_z = feat.get(f"ndvi_mean_{skey}", 0.0)
        ndvi_max_z  = feat.get(f"ndvi_max_{skey}",  0.0)
        ratio_raw   = (ndvi_max_z + 1.5) / max(ndvi_mean_z + 1.5, 0.1)
        feat[f"ratio_ndvi_{skey}"] = _zscore(ratio_raw, 1.25, 0.20)

        ndvi_all.append(veg.get("ndvi_mean", _V_REF["ndvi_mean"][si][0]))

    # ── 4. Season totals (static) ─────────────────────────────────────────────
    total = weather.get("total", {})
    feat["gdd_total"]       = _zscore(total.get("gdd_total", 1290),    1290, 200)
    feat["precip_total"]    = _zscore(total.get("precip_total", 95),    95,   65)
    feat["stress_total"]    = _zscore(total.get("stress_total", 570),   570, 180)
    feat["ndvi_max_saison"] = _zscore(max(ndvi_all) if ndvi_all else 0.65, 0.65, 0.14)

    # ── 5. Delta features (change across seasons) ─────────────────────────────
    delta_features = [
        ("ndvi_mean",       "delta_ndvi_mean"),
        ("humidity_mean",   "delta_humidity_mean"),
        ("stress_hydrique", "delta_stress_hydrique"),
    ]
    season_pairs = [("s2", "s1"), ("s3", "s2"), ("s4", "s3")]
    for raw_key, delta_prefix in delta_features:
        for sj, si in season_pairs:
            col = f"{delta_prefix}_{sj}_vs_{si}"
            diff = feat.get(f"{raw_key}_{sj}", 0.0) - feat.get(f"{raw_key}_{si}", 0.0)
            feat[col] = diff  # difference of z-scores ≈ z-score of difference

    # ── 6. Bernstein fuzzy flags (0/1) ───────────────────────────────────────
    for si, skey in enumerate(SEASONS):
        # bern_secheresse: 1 if stress_hydrique z-score is high (>0.5 std above mean)
        feat[f"bern_secheresse_{skey}"] = float(feat.get(f"stress_hydrique_{skey}", 0) > 0.5)
        # bern_ndvi_faible: 1 if ndvi_mean z-score is low (<-0.3)
        feat[f"bern_ndvi_faible_{skey}"] = float(feat.get(f"ndvi_mean_{skey}", 0) < -0.3)
        # bern_chaleur_forte: 1 if intensite_chaleur z-score is high (>0.5)
        feat[f"bern_chaleur_forte_{skey}"] = float(feat.get(f"intensite_chaleur_{skey}", 0) > 0.5)
        # bern_froid: 1 if gdd z-score is very low (<-1.0)
        feat[f"bern_froid_{skey}"] = float(feat.get(f"gdd_{skey}", 0) < -1.0)

    # ── 7. Score features (0–1) ───────────────────────────────────────────────
    # score_humid: humidity deficit score (higher = more stressed)
    for skey in ("s1", "s4"):
        h_z = feat.get(f"humidity_mean_{skey}", 0.0)
        feat[f"score_humid_{skey}"] = _sigmoid(-h_z * 0.8)  # negative z → high score

    # score_stress: water stress score
    for skey in ("s3", "s4"):
        s_z = feat.get(f"stress_hydrique_{skey}", 0.0)
        feat[f"score_stress_{skey}"] = _sigmoid(s_z * 0.8)

    # score_temp: heat score
    feat["score_temp_s3"] = _sigmoid(feat.get("intensite_chaleur_s3", 0.0) * 0.8)

    # score_global: combined stress score
    for skey in ("s3", "s4"):
        feat[f"score_global_{skey}"] = (
            feat.get(f"score_stress_{skey}", 0.5)
            * feat.get(f"score_humid_{skey if skey in ('s1','s4') else 's4'}", 0.5)
        )

    # ── 8. Interaction features ───────────────────────────────────────────────
    for skey in SEASONS:
        # inter_ndvi_evi_ratio: (ndvi - evi) relative to mean
        ndvi_z = feat.get(f"ndvi_mean_{skey}", 0.0)
        evi_z  = feat.get(f"evi_mean_{skey}", 0.0)
        feat[f"inter_ndvi_evi_ratio_{skey}"] = (ndvi_z - evi_z) * 0.7

        # inter_ndvi_gdd: ndvi quality adjusted by heat accumulation
        gdd_z = feat.get(f"gdd_{skey}", 0.0)
        feat[f"inter_ndvi_gdd_{skey}"] = ndvi_z * max(0.0, gdd_z + 1.5)

        # inter_bern_stress_ndvi: count × z-score interaction
        bern_s  = feat.get(f"bern_secheresse_{skey}", 0.0)
        bern_n  = feat.get(f"bern_ndvi_faible_{skey}", 0.0)
        feat[f"inter_bern_stress_ndvi_{skey}"] = bern_s * bern_n * max(0, -ndvi_z)

        # inter_double_stress: heat × drought combined
        bern_c  = feat.get(f"bern_chaleur_forte_{skey}", 0.0)
        feat[f"inter_double_stress_{skey}"] = bern_s * bern_c

    # ── 9. Physics flags (nearly always 0 in dataset — keep at 0) ────────────
    for col in [
        "loi_phys_froid_s1",
        "loi_phys_chaleur_s3", "loi_phys_chaleur_s4",
        "loi_phys_stress_s3",  "loi_phys_stress_s4",
    ]:
        feat[col] = 0.0

    return feat


def _extract_veg(bands: dict[str, Any], season_idx: int) -> dict[str, float]:
    """
    Derive vegetation indices for one season from the bands dict.
    Falls back to dataset means when bands are empty stubs.
    """
    ndvi_series = bands.get("ndvi_series", [])
    evi_series  = bands.get("evi_series",  [])
    evi2_series = bands.get("evi2_series", [])
    dswi_series = bands.get("dswi_series", [])
    ndwi_series = bands.get("ndwi_series", [])
    nri_series  = bands.get("nri_series",  [])
    cloud_series    = bands.get("cloud_series",     [])
    cloud_sat_series = bands.get("cloud_sat_series", [])

    def _season_vals(series: list[float]) -> list[float]:
        if len(series) >= 4:
            return [series[season_idx]]
        return series

    def _safe_stat(series: list[float], default_mean: float) -> tuple[float, float, float]:
        if not series:
            return default_mean, default_mean, 0.0
        return float(sum(series) / len(series)), float(max(series)), float(
            (max(series) - min(series)) / 2
        )

    ndvi_v = _season_vals(ndvi_series)
    evi_v  = _season_vals(evi_series)
    evi2_v = _season_vals(evi2_series)
    dswi_v = _season_vals(dswi_series)
    ndwi_v = _season_vals(ndwi_series)
    nri_v  = _season_vals(nri_series)
    cc_v   = _season_vals(cloud_series)
    ccs_v  = _season_vals(cloud_sat_series)

    ndvi_ref = _V_REF["ndvi_mean"][season_idx][0]
    evi_ref  = _V_REF["evi_mean"][season_idx][0]
    evi2_ref = _V_REF["evi2_mean"][season_idx][0]
    dswi_ref = _V_REF["dswi_mean"][season_idx][0]
    ndwi_ref = _V_REF["ndwi_mean"][season_idx][0]
    nri_ref  = _V_REF["nri_mean"][season_idx][0]

    ndvi_mean, ndvi_max, ndvi_std = _safe_stat(ndvi_v, ndvi_ref)
    evi_mean,  evi_max,  _        = _safe_stat(evi_v,  evi_ref)
    evi2_mean, evi2_max, _        = _safe_stat(evi2_v, evi2_ref)
    dswi_mean, dswi_max, _        = _safe_stat(dswi_v, dswi_ref)
    ndwi_mean, ndwi_max, _        = _safe_stat(ndwi_v, ndwi_ref)
    nri_mean,  nri_max,  _        = _safe_stat(nri_v,  nri_ref)
    cloud_mean, _, _              = _safe_stat(cc_v,  0.07)
    cloud_sat_mean, _, _          = _safe_stat(ccs_v, 0.06)

    return {
        "ndvi_mean": ndvi_mean, "ndvi_max": ndvi_max, "ndvi_std": ndvi_std,
        "evi_mean":  evi_mean,  "evi_max":  evi_max,
        "evi2_mean": evi2_mean, "evi2_max": evi2_max,
        "dswi_mean": dswi_mean, "dswi_max": dswi_max,
        "ndwi_mean": ndwi_mean, "ndwi_max": ndwi_max,
        "nri_mean":  nri_mean,  "nri_max":  nri_max,
        "cloud_cover":     cloud_mean,
        "cloud_cover_sat": cloud_sat_mean,
    }
