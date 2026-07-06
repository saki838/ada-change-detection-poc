"""FastAPI entrypoint for the internal inference microservice (port 8001).

Exposes ``POST /predict`` (JSON/base64 contract, gateway -> inference) and
``GET /health``. The Siamese U-Net is loaded once at startup so the first request
is warm; a warm-load failure is logged but never crashes the service (``mode=diff``
needs no model).
"""

from __future__ import annotations

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .diff_fallback import run_diff_predict
from .model import DEVICE, MODEL_WEIGHTS_PATH, get_model
from .predict import run_ml_predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference.main")

# Set by the startup handler and read by /health.
_MODEL_LOADED = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-load the model once so the encoder weights resolve before request 1."""
    global _MODEL_LOADED
    try:
        get_model()  # builds encoder + safe-loads head (untrained head tolerated)
        _MODEL_LOADED = bool(getattr(get_model, "weights_loaded", False))
        logger.info("Model warm-loaded (change head loaded=%s)", _MODEL_LOADED)
    except Exception as exc:  # never crash: diff mode still works
        _MODEL_LOADED = False
        logger.warning("Model warm-load failed (%s); diff mode still available", exc)
    yield


app = FastAPI(title="ADA Inference", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    t1_b64: str
    t2_b64: str
    pixel_size_m: float = 10.0
    crs: str | None = None
    geotransform: list[float] | None = None  # 6 floats, GDAL affine
    mode: Literal["ml", "diff"] = "ml"
    threshold: float = 0.5
    min_area_px: int = 20


def _decode_image(b64: str) -> np.ndarray:
    """Base64 -> HxWx3 uint8 RGB numpy array.

    Reads GeoTIFF (via rasterio, preserving band order) or ordinary PNG/JPG
    (via Pillow). Returns an RGB array; alpha is dropped downstream.
    """
    raw = base64.b64decode(b64)

    # Try rasterio first (GeoTIFF); fall back to Pillow for PNG/JPG.
    try:
        from rasterio.io import MemoryFile

        with MemoryFile(raw) as mem, mem.open() as ds:
            count = ds.count
            if count >= 3:
                arr = ds.read([1, 2, 3])  # (3,H,W)
            else:
                band = ds.read(1)
                arr = np.stack([band] * 3, axis=0)
            arr = np.transpose(arr, (1, 2, 0))  # HxWx3
            # Scale non-8-bit rasters to uint8 for a consistent model input.
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                amax = float(arr.max())
                arr = (arr / amax * 255.0) if amax > 0 else arr
                arr = arr.astype(np.uint8)
            return np.ascontiguousarray(arr)
    except Exception:
        pass

    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _encode_mask_png(mask: np.ndarray) -> str:
    """HxW uint8 (0/255) mask -> base64 PNG string."""
    from PIL import Image

    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/predict", status_code=200)
async def predict(req: PredictRequest) -> dict:
    """Core handler: decode -> validate -> branch on mode -> vectorize -> respond."""
    t0 = time.perf_counter()
    try:
        t1 = _decode_image(req.t1_b64)
        t2 = _decode_image(req.t2_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

    if t1.shape[:2] != t2.shape[:2]:
        raise HTTPException(status_code=400, detail="image size mismatch")

    try:
        if req.mode == "ml":
            mask, polygons, total_area_m2 = run_ml_predict(
                t1, t2, req.threshold, req.min_area_px,
                req.pixel_size_m, req.crs, req.geotransform,
            )
            model_name = "siamese_unet_resnet34"
        else:
            mask, polygons, total_area_m2 = run_diff_predict(
                t1, t2, req.threshold, req.min_area_px,
                req.pixel_size_m, req.crs, req.geotransform,
            )
            model_name = "image_diff"
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    height, width = int(mask.shape[0]), int(mask.shape[1])
    mask_png_b64 = _encode_mask_png(mask)
    inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "mask_png_b64": mask_png_b64,
        "width": width,
        "height": height,
        "polygons": polygons,
        "num_detections": len(polygons),
        "total_area_m2": total_area_m2,
        "model": model_name,
        "inference_ms": inference_ms,
    }


@app.get("/health")
async def health() -> dict:
    """Liveness + model-load status. Never touches the model beyond the cached flag."""
    return {
        "status": "ok",
        "model_loaded": bool(_MODEL_LOADED),
        "device": DEVICE,
        "weights": MODEL_WEIGHTS_PATH,
    }
