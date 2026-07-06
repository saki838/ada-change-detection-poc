"""Pydantic v2 request/response DTOs matching the locked API contract."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class MeResponse(BaseModel):
    id: int
    username: str
    role: str


class DetectResponse(BaseModel):
    run_id: int
    status: str = "completed"
    created_at: datetime
    mode: str
    num_detections: int
    total_area_m2: float
    mask_png_url: str        # "/api/runs/{run_id}/mask.png"
    detections_url: str      # "/api/runs/{run_id}/detections"


class RunSummary(BaseModel):
    run_id: int
    name: str | None
    status: str
    mode: str
    num_detections: int
    total_area_m2: float
    created_at: datetime


class RunListResponse(BaseModel):
    runs: list[RunSummary]
    total: int
