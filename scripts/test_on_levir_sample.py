"""
Test the real pipeline (BIT + SAM2, whatever config.yaml says) against one of
the LEVIR-CD sample pairs bundled inside the BIT_CD repo you cloned into
external/BIT_CD/samples/ — used as a stand-in until real site T1/T2 imagery
and a parcel boundary exist (Section 2's blocker in the POC doc).

LEVIR-CD samples are plain PNGs, already pixel-aligned, with no geo
referencing and no parcel boundary. This script:
  1. Picks one T1/T2 PNG pair from external/BIT_CD/samples/
  2. Wraps them as a synthetic GeoTIFF (fake but valid CRS + transform),
     exactly like scripts/run_demo.py does for its synthetic data — this is
     ONLY so Stage 0's rasterio-based alignment has something real to do.
  3. Builds a placeholder parcel boundary covering most of the image, so
     Stage 4's encroachment/setback logic has something to run against.
  4. Runs the full pipeline using whatever models are configured (real BIT +
     SAM2 if you've flipped config.yaml, dummy otherwise).

This is NOT a substitute for a real site validation run — it only proves
the real model wrappers load their weights and produce sane output.

Usage:
    python scripts/test_on_levir_sample.py
    python scripts/test_on_levir_sample.py --sample-name train_1  (pick a specific one)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.pipeline import load_config, run_pipeline

BIT_SAMPLES_DIR = pathlib.Path("external/BIT_CD/samples")


def find_sample_pair(sample_name: str | None):
    """
    BIT_CD's samples folder layout is typically:
        samples/A/<name>.png   (T1)
        samples/B/<name>.png   (T2)
    """
    a_dir = BIT_SAMPLES_DIR / "A"
    b_dir = BIT_SAMPLES_DIR / "B"
    if not a_dir.exists() or not b_dir.exists():
        raise FileNotFoundError(
            f"Expected {a_dir} and {b_dir} to exist. If BIT_CD's folder "
            "layout differs, run `dir external\\BIT_CD\\samples` (Windows) "
            "or `ls -R external/BIT_CD/samples` and tell me what's there."
        )

    a_files = sorted(a_dir.glob("*.png"))
    if not a_files:
        raise FileNotFoundError(f"No .png files found in {a_dir}")

    if sample_name:
        t1 = a_dir / f"{sample_name}.png"
        t2 = b_dir / f"{sample_name}.png"
        if not t1.exists() or not t2.exists():
            raise FileNotFoundError(f"Sample '{sample_name}' not found in A/ and B/")
        return t1, t2

    name = a_files[0].stem
    return a_dir / f"{name}.png", b_dir / f"{name}.png"


def wrap_as_geotiff(png_path: pathlib.Path, out_path: pathlib.Path, transform, crs):
    import rasterio
    from PIL import Image

    img = np.array(Image.open(png_path).convert("RGB"))
    h, w, _ = img.shape
    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=3,
        dtype=img.dtype, crs=crs, transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(img[..., band], band + 1)
    return h, w


def make_placeholder_parcel(out_path: pathlib.Path, h: int, w: int, transform, crs, gsd: float):
    """
    LEVIR-CD has no real parcel boundary, so this makes a box covering 85%
    of the image, inset from the edges — good enough to exercise Stage 4's
    encroachment/setback math without claiming to represent a real parcel.
    """
    import geopandas as gpd
    from shapely.geometry import box

    origin_x, origin_y = transform.c, transform.f
    inset = 0.075
    x0 = origin_x + inset * w * gsd
    x1 = origin_x + (1 - inset) * w * gsd
    y1 = origin_y - inset * h * gsd
    y0 = origin_y - (1 - inset) * h * gsd
    parcel = box(x0, y0, x1, y1)
    gpd.GeoDataFrame({"id": [1]}, geometry=[parcel], crs=crs).to_file(out_path, driver="GeoJSON")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="outputs/levir_sample_run")
    args = parser.parse_args()

    t1_png, t2_png = find_sample_pair(args.sample_name)
    print(f"Using sample pair:\n  T1: {t1_png}\n  T2: {t2_png}")

    from rasterio.transform import from_origin

    gsd = 0.5  # LEVIR-CD's real GSD is 0.5m/pixel — use the true value
    origin_x, origin_y = 500000.0, 4500000.0
    transform = from_origin(origin_x, origin_y, gsd, gsd)
    crs = "EPSG:32633"

    tmp_dir = pathlib.Path("data/site/_levir_sample_test")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t1_tif = tmp_dir / "t1.tif"
    t2_tif = tmp_dir / "t2.tif"
    parcel_geojson = tmp_dir / "parcel.geojson"

    h, w = wrap_as_geotiff(t1_png, t1_tif, transform, crs)
    wrap_as_geotiff(t2_png, t2_tif, transform, crs)
    make_placeholder_parcel(parcel_geojson, h, w, transform, crs, gsd)

    cfg = load_config(args.config)
    print(f"\nStage 1 model: {cfg['stage1_change_detection']['model']}")
    print(f"Stage 3 segmenter: {cfg['stage3_segment_classify']['segmenter']}")
    print(f"Stage 3 classifier: {cfg['stage3_segment_classify']['classifier']}\n")

    run_pipeline(str(t1_tif), str(t2_tif), str(parcel_geojson), args.out, cfg)

    print(f"\nDone. Check {args.out}/ for violations.geojson, overlay.png, summary.csv")
    print("Remember: the parcel boundary here is a placeholder box, not a "
          "real parcel — don't read the encroachment numbers as meaningful, "
          "just confirm the run completed without errors and the overlay "
          "highlights something sensible over the actual LEVIR-CD change.")


if __name__ == "__main__":
    main()
