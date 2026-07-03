"""
Stage 3 — Segmentation + classification.

Model A: SAM 2.1 — takes each box as a prompt -> precise polygon boundary
Model B: RT-DETR / EfficientDet — takes each box -> violation class label
Out:     Per region: a clean polygon + a violation type (illegal build, encroachment, ...)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.segment_classify import build_segmenter, build_classifier
from src.stage2_region_extraction import Region


@dataclass
class ClassifiedRegion:
    polygon_px: np.ndarray   # (N, 2) pixel coords, closed or open ring
    bbox: tuple               # (x, y, w, h) — carried through from Stage 2
    class_name: str
    confidence: float


def segment_and_classify(
    t2_rgb: np.ndarray,
    regions: list[Region],
    cfg: dict,
) -> list[ClassifiedRegion]:
    """
    cfg is the `stage3_segment_classify` block from config.yaml.
    Runs T2 (the current/"after" image) through SAM 2.1 for a precise
    polygon per box, and RT-DETR for a violation-type label per box.
    """
    segmenter = build_segmenter(cfg)
    classifier = build_classifier(cfg)

    out = []
    for region in regions:
        polygon = segmenter.segment(t2_rgb, region.bbox)
        class_name, score = classifier.classify(t2_rgb, region.bbox)
        out.append(
            ClassifiedRegion(
                polygon_px=polygon,
                bbox=region.bbox,
                class_name=class_name,
                confidence=score,
            )
        )
    return out
