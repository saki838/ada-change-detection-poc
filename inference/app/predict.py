"""ML change-detection prediction: preprocess -> forward -> threshold -> vectorize.

Turns a T1/T2 RGB array pair into a binary change mask, vectorized polygons, and
per-polygon ``area_m2`` + ``confidence``. ``mask_to_polygons`` is shared with the
non-ML ``diff_fallback`` path so both modes emit identical geometry/area/confidence.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon, mapping

from .model import DEVICE, get_model

logger = logging.getLogger("inference.predict")

# ImageNet statistics — MUST match training preprocessing (dataset.py).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 1x3xHxW float32, /255 then ImageNet-normalized."""
    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale -> 3ch
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:  # drop alpha
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(DEVICE)


def _pixel_to_crs(x: float, y: float, geotransform: list[float] | None) -> tuple[float, float]:
    """Map a pixel (col=x, row=y) to CRS coords via a 6-float GDAL affine.

    geotransform = [c, a, b, f, d, e]:
        X = c + a*col + b*row
        Y = f + d*col + e*row
    If geotransform is None, coordinates are returned in pixel space unchanged.
    """
    if geotransform is None:
        return float(x), float(y)
    c, a, b, f, d, e = geotransform
    gx = c + a * x + b * y
    gy = f + d * x + e * y
    return float(gx), float(gy)


def mask_to_polygons(
    prob: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    min_area_px: int,
    pixel_size_m: float,
    geotransform: list[float] | None,
) -> tuple[list[dict], float]:
    """Vectorize a binary mask into GeoJSON polygons with area_m2 + confidence.

    prob: HxW float probability/score map in [0,1] (used for confidence).
    mask: HxW uint8 (0/255) binary change mask.
    Returns (polygons, total_area_m2).
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[dict] = []
    total_area_m2 = 0.0
    px_area_m2 = float(pixel_size_m) ** 2

    for contour in contours:
        pixel_area = float(cv2.contourArea(contour))
        if pixel_area < min_area_px:
            continue

        # Simplify contour; epsilon proportional to perimeter.
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        pts_px = approx.reshape(-1, 2)  # (N,2) as (col,row)
        ring = [_pixel_to_crs(float(px), float(py), geotransform) for px, py in pts_px]
        shp = Polygon(ring)
        if not shp.is_valid or shp.is_empty:
            shp = shp.buffer(0)
            if shp.is_empty or shp.geom_type != "Polygon":
                continue

        # area_m2: pixel count inside contour * pixel_size^2 (flat-earth MVP).
        comp_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(comp_mask, [contour], -1, color=1, thickness=cv2.FILLED)
        pixel_count = int(comp_mask.sum())
        area_m2 = pixel_count * px_area_m2

        # confidence = mean prob over the filled pixels of this component.
        sel = comp_mask.astype(bool)
        confidence = float(prob[sel].mean()) if pixel_count > 0 else 0.0

        polygons.append(
            {
                "geometry": mapping(shp),
                "area_m2": round(area_m2, 4),
                "confidence": round(confidence, 4),
            }
        )
        total_area_m2 += area_m2

    return polygons, round(total_area_m2, 4)


def run_ml_predict(
    t1: np.ndarray,
    t2: np.ndarray,
    threshold: float,
    min_area_px: int,
    pixel_size_m: float,
    crs: str | None,
    geotransform: list[float] | None,
) -> tuple[np.ndarray, list[dict], float]:
    """Siamese U-Net forward pass -> (mask uint8 0/255, polygons, total_area_m2)."""
    model = get_model()
    x1 = preprocess(t1)
    x2 = preprocess(t2)

    with torch.no_grad():
        logits = model(x1, x2)
        prob_t = torch.sigmoid(logits)[0, 0]  # (H, W)

    prob = prob_t.cpu().numpy().astype(np.float32)
    mask = ((prob >= threshold).astype(np.uint8)) * 255

    polygons, total_area_m2 = mask_to_polygons(
        prob, mask, threshold, min_area_px, pixel_size_m, geotransform
    )
    return mask, polygons, total_area_m2
