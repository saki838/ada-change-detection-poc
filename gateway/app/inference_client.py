"""Async httpx wrapper that base64-encodes T1/T2 and POSTs to inference /predict."""
from __future__ import annotations

import base64

import httpx

from app.config import get_settings


async def call_predict(
    t1_bytes: bytes,
    t2_bytes: bytes,
    *,
    pixel_size_m: float = 10.0,
    crs: str | None = None,
    geotransform: list[float] | None = None,
    mode: str = "ml",
    threshold: float = 0.5,
    min_area_px: int = 20,
) -> dict:
    """Call POST {INFERENCE_URL}/predict and return the parsed JSON response.

    Raises httpx.HTTPStatusError on non-2xx and httpx.HTTPError on transport
    failure so the caller can translate them into 400/502 responses.
    """
    settings = get_settings()
    body = {
        "t1_b64": base64.b64encode(t1_bytes).decode("ascii"),
        "t2_b64": base64.b64encode(t2_bytes).decode("ascii"),
        "pixel_size_m": pixel_size_m,
        "crs": crs,
        "geotransform": geotransform,
        "mode": mode,
        "threshold": threshold,
        "min_area_px": min_area_px,
    }
    url = f"{settings.inference_url.rstrip('/')}/predict"
    async with httpx.AsyncClient(timeout=settings.inference_timeout_s) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        return resp.json()
