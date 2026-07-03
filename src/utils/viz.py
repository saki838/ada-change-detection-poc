"""Visualization helpers — overlay a change mask / violations on the T2 image."""
from __future__ import annotations

import numpy as np


def make_overlay(base_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    """
    base_rgb: (H, W, 3) uint8
    mask: (H, W) bool or {0,1}
    Returns an (H, W, 3) uint8 image with `mask` regions tinted `color`.
    """
    out = base_rgb.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boxes_and_labels(base_rgb: np.ndarray, regions: list) -> np.ndarray:
    """
    Draw bounding boxes + class labels on top of an RGB image for a quick
    visual sanity check. `regions` is a list of dicts with keys
    'bbox' (x, y, w, h) and 'class_name'.
    """
    import cv2

    out = base_rgb.copy()
    for r in regions:
        x, y, w, h = r["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out, r.get("class_name", "?"), (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return out
