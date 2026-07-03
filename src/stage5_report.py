"""
Stage 5 — Report generation.

Tool: Custom Python script
Out:  GeoJSON of violations + overlay PNG (mask on drone image) + summary table
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from src.stage4_measurement import Violation
from src.utils.viz import make_overlay


def write_report(
    violations: list[Violation],
    t2_rgb: np.ndarray,
    change_mask: np.ndarray,
    crs,
    out_dir: str,
    overlay_alpha: float = 0.45,
    overlay_mask_color: tuple = (255, 0, 0),
) -> dict:
    """
    Writes:
        <out_dir>/violations.geojson
        <out_dir>/overlay.png
        <out_dir>/summary.csv
    Returns a dict of the written paths, for the pipeline/CLI to print.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson_path = out_dir / "violations.geojson"
    _write_geojson(violations, crs, geojson_path)

    overlay_path = out_dir / "overlay.png"
    _write_overlay(t2_rgb, change_mask, overlay_path, overlay_alpha, overlay_mask_color)

    summary_path = out_dir / "summary.csv"
    _write_summary(violations, summary_path)

    return {
        "geojson": str(geojson_path),
        "overlay_png": str(overlay_path),
        "summary_csv": str(summary_path),
    }


def _write_geojson(violations: list[Violation], crs, path: pathlib.Path):
    import geopandas as gpd

    if not violations:
        gpd.GeoDataFrame(
            columns=["class_name", "confidence", "area_m2", "encroaches_parcel",
                     "encroachment_area_m2", "setback_violation", "min_setback_m",
                     "red_zone_overlap", "red_zone_overlap_m2", "geometry"],
            geometry="geometry", crs=crs,
        ).to_file(path, driver="GeoJSON")
        return

    gdf = gpd.GeoDataFrame(
        {
            "class_name": [v.class_name for v in violations],
            "confidence": [v.confidence for v in violations],
            "area_m2": [v.area_m2 for v in violations],
            "encroaches_parcel": [v.encroaches_parcel for v in violations],
            "encroachment_area_m2": [v.encroachment_area_m2 for v in violations],
            "setback_violation": [v.setback_violation for v in violations],
            "min_setback_m": [v.min_setback_m for v in violations],
            "red_zone_overlap": [v.red_zone_overlap for v in violations],
            "red_zone_overlap_m2": [v.red_zone_overlap_m2 for v in violations],
            "geometry": [v.polygon_world for v in violations],
        },
        geometry="geometry",
        crs=crs,
    )
    gdf.to_file(path, driver="GeoJSON")


def _write_overlay(t2_rgb, change_mask, path, alpha, color):
    from PIL import Image

    overlay = make_overlay(t2_rgb, change_mask, color=color, alpha=alpha)
    Image.fromarray(overlay).save(path)


def _write_summary(violations: list[Violation], path: pathlib.Path):
    import pandas as pd

    if not violations:
        pd.DataFrame(columns=[
            "class_name", "confidence", "area_m2", "encroaches_parcel",
            "encroachment_area_m2", "setback_violation", "min_setback_m",
            "red_zone_overlap", "red_zone_overlap_m2",
        ]).to_csv(path, index=False)
        return

    df = pd.DataFrame([
        {
            "class_name": v.class_name,
            "confidence": round(v.confidence, 3),
            "area_m2": round(v.area_m2, 2),
            "encroaches_parcel": v.encroaches_parcel,
            "encroachment_area_m2": round(v.encroachment_area_m2, 2),
            "setback_violation": v.setback_violation,
            "min_setback_m": round(v.min_setback_m, 2) if v.min_setback_m is not None else None,
            "red_zone_overlap": v.red_zone_overlap,
            "red_zone_overlap_m2": round(v.red_zone_overlap_m2, 2),
        }
        for v in violations
    ])
    df.to_csv(path, index=False)
