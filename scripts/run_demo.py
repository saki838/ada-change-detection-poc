"""
End-to-end smoke test: generates a synthetic T1/T2 GeoTIFF pair + a parcel
boundary, then runs the full 6-stage pipeline in dummy mode (no GPU, no
downloaded weights required). This is the fastest way to prove the pipeline
plumbing works before wiring in real BIT / SAM2.1 / RT-DETR weights.

Usage:
    python scripts/run_demo.py --dummy
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.pipeline import load_config, force_dummy_mode, run_pipeline


def make_synthetic_pair(tmp_dir: pathlib.Path, size: int = 512):
    """
    Creates:
        tmp_dir/t1_baseline.tif   — flat "empty lot" scene
        tmp_dir/t2_current.tif    — same scene + a synthetic "new structure"
        tmp_dir/parcel.geojson    — parcel boundary that the new structure
                                    partially encroaches past
    All georeferenced to a fake local UTM-like CRS so Stage 0/4 run for real.
    """
    import rasterio
    from rasterio.transform import from_origin
    import geopandas as gpd
    from shapely.geometry import Polygon, box

    rng = np.random.default_rng(0)
    gsd = 0.10  # 10cm/pixel
    origin_x, origin_y = 500000.0, 4500000.0  # arbitrary UTM-like origin
    transform = from_origin(origin_x, origin_y, gsd, gsd)
    crs = "EPSG:32633"  # UTM zone 33N — any projected CRS works for the demo

    # Baseline: grass-green flat lot with mild noise texture.
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[..., 0] = 90 + rng.integers(-5, 5, (size, size))
    base[..., 1] = 140 + rng.integers(-5, 5, (size, size))
    base[..., 2] = 80 + rng.integers(-5, 5, (size, size))

    # Current: same lot, plus a synthetic gray "structure" rectangle that
    # crosses the parcel's east boundary (the encroachment we want detected).
    # Coordinates are fractions of `size` so this scales to any image size.
    current = base.copy()
    y0, y1 = int(0.39 * size), int(0.625 * size)
    x0, x1 = int(0.586 * size), int(0.898 * size)
    current[y0:y1, x0:x1] = [150, 150, 150]

    t1_path = tmp_dir / "t1_baseline.tif"
    t2_path = tmp_dir / "t2_current.tif"
    _write_geotiff(t1_path, base, transform, crs)
    _write_geotiff(t2_path, current, transform, crs)

    # Parcel boundary in world coords: a box that the structure pokes out of
    # on the right/east edge, to exercise the encroachment/setback logic.
    parcel_x0, parcel_y1 = origin_x, origin_y  # top-left of raster
    parcel_x1 = origin_x + 0.82 * size * gsd    # cuts through the structure
    parcel_y0 = origin_y - size * gsd
    parcel = box(parcel_x0, parcel_y0, parcel_x1, parcel_y1)
    gpd.GeoDataFrame({"id": [1]}, geometry=[parcel], crs=crs).to_file(
        tmp_dir / "parcel.geojson", driver="GeoJSON"
    )

    return t1_path, t2_path, tmp_dir / "parcel.geojson"


def _write_geotiff(path: pathlib.Path, rgb: np.ndarray, transform, crs):
    import rasterio

    h, w, _ = rgb.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3,
        dtype=rgb.dtype, crs=crs, transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(rgb[..., band], band + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", default=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="outputs/demo_run")
    args = parser.parse_args()

    tmp_dir = pathlib.Path("data/site/_synthetic_demo")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t1_path, t2_path, parcel_path = make_synthetic_pair(tmp_dir)

    cfg = load_config(args.config)
    cfg = force_dummy_mode(cfg)

    run_pipeline(str(t1_path), str(t2_path), str(parcel_path), args.out, cfg)

    print(f"\nDemo complete. Check {args.out}/ for violations.geojson, "
          f"overlay.png and summary.csv")


if __name__ == "__main__":
    main()
