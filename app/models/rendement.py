from typing import Any

from pydantic import BaseModel


class Parcelle(BaseModel):
    id: str
    polygone: dict[str, Any]
    date_plantation: str
    variete: str


class RendementRequest(BaseModel):
    parcelle: Parcelle
    date_prediction: str


class ShapFeature(BaseModel):
    feature: str
    impact: float  # positive or negative float


class IncertitudeInfo(BaseModel):
    niveau: str  # "NORMALE" | "HAUTE"
    sigma_log: float


class RendementResponse(BaseModel):
    tonnage_predit_t: float
    intervalle_confiance_95: list[float]  # [min, max]
    top_features_shap: list[ShapFeature]
    date_recolte_estimee: str
    incertitude: IncertitudeInfo | None = None
