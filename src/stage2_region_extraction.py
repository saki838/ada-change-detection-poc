"""
Stage 2 — Region extraction.

Tool: OpenCV (contours -> bounding boxes)
In:   Binary change mask
Out:  List of bounding boxes, one per discrete change region (drops noise/specks)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Region:
    bbox: tuple            # (x, y, w, h) in pixel coords
    contour: np.ndarray     # raw (N, 1, 2) contour points, for reference
    area_px: float


def extract_regions(
    mask: np.ndarray,
    min_area_px: int = 150,
    morphology_kernel: int = 5,
    approx_epsilon: float = 2.0,
) -> list[Region]:
    import cv2

    mask_u8 = (mask.astype(np.uint8)) * 255

    # Clean speckle noise: opening removes small isolated blobs, closing
    # fills small holes inside real regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        simplified = cv2.approxPolyDP(c, approx_epsilon, closed=True)
        regions.append(Region(bbox=(x, y, w, h), contour=simplified, area_px=area))

    return regions
