from typing import Any, Literal

from pydantic import BaseModel


class Oliveraie(BaseModel):
    id: str
    polygone: dict[str, Any]
    systeme: Literal["extensif", "intensif", "hyper_intensif"]


class AnomalieRequest(BaseModel):
    oliveraie: Oliveraie
    date: str


class AnomalieResponse(BaseModel):
    statut: Literal["vert", "orange", "rouge"]
    anomaly_score: float  # unbounded; e.g. 2.4 = 240% deviation
    ndvi_observe: list[float]  # time-series over sliding window
    ndvi_attendu: list[float]  # baseline time-series
    explication: str
    recommandation: str
