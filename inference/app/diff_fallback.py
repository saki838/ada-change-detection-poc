"""Non-ML ``mode=diff`` fallback: change detection by raw image differencing.

Needs no trained checkpoint, so the demo runs meaningfully even before a change
head is trained. Reuses ``mask_to_polygons`` from ``predict`` so the vectorization,
area, and confidence semantics are identical to the ML path. Returns the same
``(mask, polygons, total_area_m2)`` tuple shape so ``main.py`` treats both modes
identically.
"""

from __future__ import annotations

import cv2
import numpy as np

from .predict import mask_to_polygons


def run_diff_predict(
    t1: np.ndarray,
    t2: np.ndarray,
    threshold: float,
    min_area_px: int,
    pixel_size_m: float,
    crs: str | None,
    geotransform: list[float] | None,
) -> tuple[np.ndarray, list[dict], float]:
    """Grayscale abs-diff -> blur -> normalize -> binarize -> vectorize."""
    a = np.asarray(t1)
    b = np.asarray(t2)
    if a.ndim == 3 and a.shape[2] >= 3:
        g1 = cv2.cvtColor(a[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        g1 = a.astype(np.uint8)
    if b.ndim == 3 and b.shape[2] >= 3:
        g2 = cv2.cvtColor(b[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        g2 = b.astype(np.uint8)

    diff = cv2.absdiff(g1, g2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # Normalize diff to [0,1] as a pseudo-probability map (confidence source).
    dmax = float(diff.max())
    prob = (diff.astype(np.float32) / dmax) if dmax > 0 else np.zeros_like(diff, dtype=np.float32)

    # Binarize: Otsu gives a data-driven cut; the caller threshold (0..1) acts as
    # an additional floor on the normalized diff.
    _, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = (prob >= float(threshold)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(otsu, floor)

    polygons, total_area_m2 = mask_to_polygons(
        prob, mask, threshold, min_area_px, pixel_size_m, geotransform
    )
    return mask, polygons, total_area_m2
