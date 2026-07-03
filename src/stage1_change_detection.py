"""
Stage 1 — Change detection (the core model).

Model: BIT (primary, lightweight) · ChangeFormer (higher-accuracy alternative)
In:    Aligned (T1, T2) pair
Out:   Binary change mask — pixels that changed between T1 and T2
"""
from __future__ import annotations

import numpy as np

from src.models.change_detector import build_change_detector


def detect_change(t1_rgb: np.ndarray, t2_rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """
    cfg is the `stage1_change_detection` block from config.yaml.
    Returns a binary (H, W) uint8 mask, {0, 1}.
    """
    detector = build_change_detector(cfg)
    mask = detector.predict(t1_rgb, t2_rgb)
    assert mask.shape == t1_rgb.shape[:2], (
        f"Change mask shape {mask.shape} doesn't match input {t1_rgb.shape[:2]}"
    )
    return mask
