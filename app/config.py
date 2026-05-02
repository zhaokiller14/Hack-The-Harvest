from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gee_project: str = ""
    sentinel_date_start: str = "2024-05-01"
    sentinel_date_end: str = "2024-06-30"
    sentinel_cloud_pct: int = 40
    max_polygon_km2: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
