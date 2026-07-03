"""
Orchestrator — wires Stage 0 through Stage 5 together exactly as specified
in Section 3 of the POC doc:

    Stage 0 Align -> Stage 1 Change detection -> Stage 2 Region extraction
    -> Stage 3 Segment + classify -> Stage 4 Measurement -> Stage 5 Report

Usage:
    python -m src.pipeline --t1 T1.tif --t2 T2.tif --parcel parcel.geojson \
        --out outputs/run_001 [--config config.yaml] [--dummy]
"""
from __future__ import annotations

import argparse
import copy

import yaml

from src.stage0_align import align_rasters
from src.stage1_change_detection import detect_change
from src.stage2_region_extraction import extract_regions
from src.stage3_segment_classify import segment_and_classify
from src.stage4_measurement import measure_violations
from src.stage5_report import write_report


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def force_dummy_mode(cfg: dict) -> dict:
    """Swap every model-backed stage to its no-weights stand-in, for smoke tests."""
    cfg = copy.deepcopy(cfg)
    cfg["stage1_change_detection"]["model"] = "dummy_diff"
    cfg["stage3_segment_classify"]["segmenter"] = "dummy_contour"
    cfg["stage3_segment_classify"]["classifier"] = "dummy_rule"
    return cfg


def run_pipeline(
    t1_path: str,
    t2_path: str,
    parcel_path: str | None,
    out_dir: str,
    cfg: dict,
    red_zone_path: str | None = None,
) -> dict:
    print("[Stage 0] Aligning T1/T2 rasters...")
    aligned = align_rasters(
        t1_path, t2_path,
        target_resolution_m=cfg["stage0_align"]["target_resolution_m"],
        resampling_method=cfg["stage0_align"]["resampling_method"],
    )

    print("[Stage 1] Running change detection...")
    change_mask = detect_change(aligned.t1_rgb, aligned.t2_rgb, cfg["stage1_change_detection"])
    print(f"  -> {change_mask.sum()} changed pixels")

    print("[Stage 2] Extracting change regions...")
    regions = extract_regions(
        change_mask,
        min_area_px=cfg["stage2_region_extraction"]["min_area_px"],
        morphology_kernel=cfg["stage2_region_extraction"]["morphology_kernel"],
        approx_epsilon=cfg["stage2_region_extraction"]["approx_epsilon"],
    )
    print(f"  -> {len(regions)} candidate regions")

    print("[Stage 3] Segmenting + classifying regions...")
    classified = segment_and_classify(aligned.t2_rgb, regions, cfg["stage3_segment_classify"])

    print("[Stage 4] Measuring violations...")
    violations = measure_violations(
        classified,
        transform=aligned.transform,
        crs=aligned.crs,
        parcel_boundary_path=parcel_path,
        red_zone_boundary_path=red_zone_path,
        setback_distance_m=cfg["stage4_measurement"]["setback_distance_m"],
    )
    print(f"  -> {len(violations)} measured violations")

    print("[Stage 5] Writing report...")
    paths = write_report(
        violations,
        t2_rgb=aligned.t2_rgb,
        change_mask=change_mask,
        crs=aligned.crs,
        out_dir=out_dir,
        overlay_alpha=cfg["stage5_report"]["overlay_alpha"],
        overlay_mask_color=tuple(cfg["stage5_report"]["overlay_mask_color"]),
    )
    print(f"  -> {paths}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="ADA change-detection POC pipeline")
    parser.add_argument("--t1", required=True, help="Baseline / approved-plan raster")
    parser.add_argument("--t2", required=True, help="Current drone raster")
    parser.add_argument("--parcel", default=None, help="Parcel boundary GeoJSON/SHP")
    parser.add_argument("--red-zone", default=None, help="Optional red-zone boundary GeoJSON/SHP")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--dummy", action="store_true",
        help="Use no-weights stand-in models everywhere (smoke test / no GPU)",
    )
    parser.add_argument("--weights", default=None, help="Override stage1 weights_path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dummy:
        cfg = force_dummy_mode(cfg)
    if args.weights:
        cfg["stage1_change_detection"]["weights_path"] = args.weights

    run_pipeline(args.t1, args.t2, args.parcel, args.out, cfg, red_zone_path=args.red_zone)


if __name__ == "__main__":
    main()
