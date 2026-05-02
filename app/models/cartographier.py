from typing import Any

from pydantic import BaseModel


class CartographierRequest(BaseModel):
    polygone_perimetre: dict[str, Any]  # GeoJSON polygon
    date: str


class OliveraieFeature(BaseModel):
    polygone: dict[str, Any]
    systeme: str  # "extensif" | "intensif"
    confiance: float
    surface_ha: float


class CartographierStats(BaseModel):
    total_oliveraies: int
    surface_totale_ha: float
    repartition: dict[str, int]  # {"extensif": n, "intensif": n}


class CartographierResponse(BaseModel):
    oliveraies: list[OliveraieFeature]
    stats: CartographierStats
