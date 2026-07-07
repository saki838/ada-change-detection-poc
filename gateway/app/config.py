"""Typed application settings loaded from the environment (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    jwt_secret: str = "change-me-to-a-64-char-random-hex-string"
    jwt_alg: str = "HS256"
    jwt_expires_in: int = 3600                             # env JWT_EXPIRES_IN
    database_url: str = "postgresql+psycopg://ada:ada@db:5432/ada"  # env DATABASE_URL
    inference_url: str = "http://inference:8001"           # env INFERENCE_URL
    inference_timeout_s: int = 120                          # env INFERENCE_TIMEOUT_S
    image_store_dir: str = "/data/images"                  # env IMAGE_STORE_DIR
    # NoDecode: skip pydantic-settings' JSON source-decode so the plain/comma-separated
    # CORS_ORIGINS string reaches _split_cors below instead of failing JSON parse.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]  # env CORS_ORIGINS

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v):
        # CORS_ORIGINS arrives from compose/.env as a plain (comma-separated) string,
        # e.g. "http://localhost:5173". Accept that as well as a real list / JSON.
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):
                return v  # let pydantic parse the JSON list form
            return [o.strip() for o in s.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """LRU-cached singleton so the environment is read exactly once."""
    return Settings()
