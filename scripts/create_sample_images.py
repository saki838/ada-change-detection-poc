"""
Generate synthetic sample images for the ADA change-detection POC.

Creates realistic-looking T1/T2 image pairs with various change types:
  - new_construction: A building/structure appears between T1 and T2
  - demolition: A structure disappears between T1 and T2
  - vegetation_clearance: Green area turns brown/bare
  - horizontal_expansion: An existing structure expands
  - unauthorized_paving: A natural surface gets paved
  - encroachment: New construction crosses a parcel boundary

Output layout:
  data/site/samples/{scenario}/{sample_id}/
      t1.tif         — Baseline GeoTIFF (time 1)
      t2.tif         — Current GeoTIFF (time 2, with changes)
      parcel.geojson — Optional parcel boundary for encroachment tests
      label.png      — Ground-truth change mask (white = change, black = no change)

Usage:
    python scripts/create_sample_images.py [--count 3] [--size 512] [--out data/site/samples]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow is required. Install: pip install Pillow")
    sys.exit(1)

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    print("rasterio is required. Install: pip install rasterio")
    sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, box
except ImportError:
    print("geopandas + shapely are required. Install: pip install geopandas shapely")
    sys.exit(1)


# ── Colour palettes ───────────────────────────────────────────────

COLORS = {
    "grass":       np.array([90,  140, 80],  dtype=np.uint8),
    "dark_grass":  np.array([70,  110, 60],  dtype=np.uint8),
    "bare_soil":   np.array([160, 135, 100], dtype=np.uint8),
    "concrete":    np.array([180, 180, 175], dtype=np.uint8),
    "roof_red":    np.array([180, 80,  70],  dtype=np.uint8),
    "roof_grey":   np.array([150, 150, 155], dtype=np.uint8),
    "roof_blue":   np.array([80,  120, 170], dtype=np.uint8),
    "asphalt":     np.array([100, 100, 105], dtype=np.uint8),
    "shadow":      np.array([40,  40,  45],  dtype=np.uint8),
    "tree":        np.array([50,  110, 40],  dtype=np.uint8),
    "water":       np.array([60,  100, 140], dtype=np.uint8),
    "fence":       np.array([130, 130, 120], dtype=np.uint8),
    "white":       np.array([255, 255, 255], dtype=np.uint8),
}


def _add_noise(base: np.ndarray, intensity: int = 5) -> np.ndarray:
    """Add mild Gaussian noise to an image."""
    noise = np.random.default_rng().integers(-intensity, intensity + 1, base.shape, dtype=np.int16)
    noisy = base.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray):
    """Fill a rectangle in-place."""
    img[y0:y1, x0:x1] = color.astype(np.uint8)


def _draw_rect_outline(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                       color: np.ndarray, thickness: int = 2):
    """Draw an outlined rectangle in-place."""
    img[y0:y0 + thickness, x0:x1] = color.astype(np.uint8)
    img[y1 - thickness:y1, x0:x1] = color.astype(np.uint8)
    img[y0:y1, x0:x0 + thickness] = color.astype(np.uint8)
    img[y0:y1, x1 - thickness:x1] = color.astype(np.uint8)


def _write_geotiff(path: pathlib.Path, rgb: np.ndarray, transform, crs: str):
    """Write an RGB numpy array as a 3-band GeoTIFF."""
    h, w, _ = rgb.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3,
        dtype=rgb.dtype, crs=crs, transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(rgb[..., band], band + 1)


# ── Scene Generators ──────────────────────────────────────────────

def generate_new_construction(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A building appears between T1 and T2."""
    rng = np.random.default_rng(42)

    # T1: Grass field with some trees
    t1 = np.zeros((size, size, 3), dtype=np.uint8)
    t1[:] = COLORS["grass"] + rng.integers(-3, 4, (size, size, 3))
    # Some trees
    for _ in range(8):
        cx, cy = rng.integers(50, size - 50), rng.integers(50, size - 50)
        r = rng.integers(8, 20)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        t1[mask] = COLORS["tree"] + rng.integers(-3, 4, (3,))

    # T2: Same field + a new building + construction debris
    t2 = t1.copy()
    bx0, by0 = int(size * 0.35), int(size * 0.40)
    bx1, by1 = int(size * 0.65), int(size * 0.72)
    _draw_rect(t2, bx0, by0, bx1, by1, COLORS["roof_red"])
    # Shadow
    shadow_x0, shadow_y0 = bx0 + 5, by0 + 5
    shadow_x1, shadow_y1 = bx1 + 12, by1 + 12
    _draw_rect(t2, shadow_x0, shadow_y0,
               min(shadow_x1, size), min(shadow_y1, size),
               COLORS["shadow"])
    # Re-draw building on top of shadow
    _draw_rect(t2, bx0, by0, bx1, by1, COLORS["roof_red"])
    # Construction debris around it
    _draw_rect(t2, bx0 - 8, by1, bx1 + 8, by1 + 6, COLORS["bare_soil"])

    # Change mask
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[by0:by1, bx0:bx1] = 255
    mask[by1:by1 + 6, bx0 - 8:bx1 + 8] = 255

    return _add_noise(t1, 3), _add_noise(t2, 3), mask


def generate_demolition(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A building disappears between T1 and T2."""
    rng = np.random.default_rng(43)

    # T1: Field with a building
    t1 = np.zeros((size, size, 3), dtype=np.uint8)
    t1[:] = COLORS["grass"] + rng.integers(-3, 4, (size, size, 3))

    bx0, by0 = int(size * 0.30), int(size * 0.35)
    bx1, by1 = int(size * 0.60), int(size * 0.65)
    _draw_rect(t1, bx0, by0, bx1, by1, COLORS["roof_grey"])
    _draw_rect(t1, bx0 + 5, by0 + 5, bx1 - 5, by1 - 5, COLORS["roof_blue"])
    # Shadow
    _draw_rect(t1, bx0 + 8, by0 + 8, bx1 + 15, by1 + 15, COLORS["shadow"])
    _draw_rect(t1, bx0, by0, bx1, by1, COLORS["roof_grey"])
    _draw_rect(t1, bx0 + 5, by0 + 5, bx1 - 5, by1 - 5, COLORS["roof_blue"])

    # T2: Rubble pile where building was
    t2 = t1.copy()
    _draw_rect(t2, bx0 - 5, by0 - 5, bx1 + 5, by1 + 5, COLORS["bare_soil"])
    # Some debris texture
    for _ in range(30):
        dx, dy = rng.integers(bx0 - 5, bx1 + 5), rng.integers(by0 - 5, by1 + 5)
        t2[dy, dx] = COLORS["concrete"]

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[by0 - 5:by1 + 5, bx0 - 5:bx1 + 5] = 255

    return _add_noise(t1, 3), _add_noise(t2, 3), mask


def generate_vegetation_clearance(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Green area cleared to bare soil between T1 and T2."""
    rng = np.random.default_rng(44)

    t1 = np.zeros((size, size, 3), dtype=np.uint8)
    t1[:] = COLORS["grass"] + rng.integers(-3, 4, (size, size, 3))

    # Dense vegetation patch
    cx, cy = int(size * 0.6), int(size * 0.5)
    r = int(size * 0.25)
    yy, xx = np.ogrid[:size, :size]
    veg_mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    t1[veg_mask] = COLORS["tree"] + rng.integers(-5, 6, (3,))

    # T2: That patch is gone, replaced by bare soil
    t2 = t1.copy()
    t2[veg_mask] = COLORS["bare_soil"] + rng.integers(-5, 6, (3,))
    # Some dirt road tracks through it
    for y in range(cy - r + 10, cy + r - 10, 4):
        x_offset = int((r - abs(y - cy)) * 0.5)
        t2[y, cx - 8 + x_offset:cx + 8 - x_offset] = COLORS["bare_soil"] + rng.integers(-3, 4, (3,))

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[veg_mask] = 255

    return _add_noise(t1, 2), _add_noise(t2, 3), mask


def generate_horizontal_expansion(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An existing building expands to the side."""
    rng = np.random.default_rng(45)

    t1 = np.zeros((size, size, 3), dtype=np.uint8)
    t1[:] = COLORS["grass"] + rng.integers(-3, 4, (size, size, 3))

    # Original building
    bx0, by0 = int(size * 0.25), int(size * 0.35)
    bx1, by1 = int(size * 0.50), int(size * 0.70)
    _draw_rect(t1, bx0, by0, bx1, by1, COLORS["roof_red"])
    _draw_rect(t1, bx0 + 5, by0 + 5, bx1 - 5, by1 - 5, COLORS["roof_grey"])

    t2 = t1.copy()
    # Extension to the right
    ext_x0, ext_x1 = bx1, int(size * 0.72)
    _draw_rect(t2, ext_x0, by0 + 10, ext_x1, by1 - 10, COLORS["roof_red"])
    _draw_rect(t2, ext_x0 + 3, by0 + 13, ext_x1 - 3, by1 - 13, COLORS["roof_grey"])

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[by0 + 10:by1 - 10, ext_x0:ext_x1] = 255

    return _add_noise(t1, 3), _add_noise(t2, 3), mask


def generate_unauthorized_paving(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A natural area gets paved over for parking."""
    rng = np.random.default_rng(46)

    t1 = np.zeros((size, size, 3), dtype=np.uint8)
    t1[:] = COLORS["grass"] + rng.integers(-3, 4, (size, size, 3))
    # A few existing trees
    for _ in range(4):
        cx, cy = rng.integers(30, size - 30), rng.integers(30, size - 30)
        r = rng.integers(6, 15)
        yy, xx = np.ogrid[:size, :size]
        tree_mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        t1[tree_mask] = COLORS["tree"]

    # T2: Large paved area in the center
    t2 = t1.copy()
    px0, py0 = int(size * 0.25), int(size * 0.30)
    px1, py1 = int(size * 0.75), int(size * 0.80)
    _draw_rect(t2, px0, py0, px1, py1, COLORS["asphalt"])
    # Parking lines
    for y in range(py0 + 10, py1 - 10, 20):
        for x in range(px0 + 10, px1 - 10, 15):
            _draw_rect(t2, x, y, x + 10, y + 3, COLORS["white"])

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[py0:py1, px0:px1] = 255

    return _add_noise(t1, 3), _add_noise(t2, 3), mask


def generate_encroachment(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Polygon]:
    """New construction crosses a parcel boundary."""
    rng = np.random.default_rng(47)

    t1 = np.zeros((size, size, 3), dtype=np.uint8)
    t1[:] = COLORS["grass"] + rng.integers(-3, 4, (size, size, 3))

    # Parcel boundary (box covering ~70% of the image, left side)
    parcel = box(
        int(size * 0.05), int(size * 0.05),
        int(size * 0.65), int(size * 0.85),
    )

    # T2: A building straddles the parcel boundary
    t2 = t1.copy()
    bx0, by0 = int(size * 0.45), int(size * 0.35)
    bx1, by1 = int(size * 0.78), int(size * 0.70)
    _draw_rect(t2, bx0, by0, bx1, by1, COLORS["roof_blue"])
    _draw_rect(t2, bx0 + 5, by0 + 5, bx1 - 5, by1 - 5, COLORS["roof_grey"])

    # Draw parcel boundary line on the image for visualization
    minx, miny, maxx, maxy = map(int, parcel.bounds)
    _draw_rect_outline(t2, minx, miny, maxx, maxy, COLORS["fence"], thickness=3)

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[by0:by1, bx0:bx1] = 255

    return _add_noise(t1, 3), _add_noise(t2, 3), mask, parcel


# ── Main Generator ────────────────────────────────────────────────

SCENARIO_GENERATORS = {
    "new_construction":     generate_new_construction,
    "demolition":            generate_demolition,
    "vegetation_clearance": generate_vegetation_clearance,
    "horizontal_expansion": generate_horizontal_expansion,
    "unauthorized_paving":  generate_unauthorized_paving,
    "encroachment":         generate_encroachment,
}


def generate_samples(
    out_dir: str,
    count_per_scenario: int = 3,
    size: int = 512,
    crs: str = "EPSG:32643",
    gsd: float = 0.5,
):
    """Generate synthetic sample image pairs for each scenario."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    origin_x, origin_y = 500000.0, 4500000.0
    transform = from_origin(origin_x, origin_y, gsd, gsd)

    summary_lines = []

    for scenario_name, generator_fn in SCENARIO_GENERATORS.items():
        for sample_id in range(1, count_per_scenario + 1):
            sample_dir = out / scenario_name / f"sample_{sample_id:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            if scenario_name == "encroachment":
                t1, t2, label, parcel_px = generator_fn(size)
                # Convert pixel coords to real-world coords to match GeoTIFF CRS
                minx, miny, maxx, maxy = parcel_px.bounds
                parcel_world = box(
                    origin_x + minx * gsd,
                    origin_y - maxy * gsd,
                    origin_x + maxx * gsd,
                    origin_y - miny * gsd,
                )
                gdf = gpd.GeoDataFrame(
                    {"id": [1], "scenario": [scenario_name]},
                    geometry=[parcel_world],
                    crs=crs,
                )
                parcel_path = sample_dir / "parcel.geojson"
                gdf.to_file(parcel_path, driver="GeoJSON")
            else:
                t1, t2, label = generator_fn(size)

            t1_path = sample_dir / "t1.tif"
            t2_path = sample_dir / "t2.tif"
            label_path = sample_dir / "label.png"

            _write_geotiff(t1_path, t1, transform, crs)
            _write_geotiff(t2_path, t2, transform, crs)
            Image.fromarray(label).save(label_path)

            # Create a quick-look composite
            composite = t2.copy()
            composite[..., 0] = np.where(label > 0, 255, composite[..., 0])
            composite_path = sample_dir / "composite_preview.png"
            Image.fromarray(composite).save(composite_path)

            line = (f"  ✅ {scenario_name}/sample_{sample_id:03d}  "
                    f"({t1.shape[1]}x{t1.shape[0]})  "
                    f"{'with parcel' if scenario_name == 'encroachment' else ''}")
            summary_lines.append(line)
            print(line)

    return summary_lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample images for ADA change-detection POC",
    )
    parser.add_argument(
        "--out", default="data/site/samples",
        help="Output directory (default: data/site/samples)",
    )
    parser.add_argument(
        "--count", type=int, default=3,
        help="Number of samples per scenario (default: 3)",
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Image size in pixels (default: 512)",
    )
    parser.add_argument(
        "--gsd", type=float, default=0.5,
        help="Ground sample distance in metres (default: 0.5)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏗️  ADA Change-Detection Sample Image Generator")
    print("=" * 60)
    print(f"\n📁 Output:  {args.out}")
    print(f"🖼️  Size:    {args.size}x{args.size} px")
    print(f"📏 GSD:     {args.gsd}m/px")
    print(f"🔢 Samples: {args.count} per scenario")
    print(f"🎭 Scenarios: {', '.join(SCENARIO_GENERATORS.keys())}")
    print()

    lines = generate_samples(args.out, args.count, args.size, gsd=args.gsd)

    print()
    print("=" * 60)
    print(f"🎉 Generated {len(lines)} sample pairs successfully!")
    print("=" * 60)
    print()
    print("📁 Output structure:")
    print(f"   {args.out}/{{scenario}}/sample_{{nnn}}/")
    print(f"       ├── t1.tif              (baseline GeoTIFF)")
    print(f"       ├── t2.tif              (current GeoTIFF)")
    print(f"       ├── label.png           (change mask)")
    print(f"       ├── composite_preview.png (T2 + change overlay)")
    print(f"       └── parcel.geojson      (only for encroachment)")
    print()
    print("💡 To use with the pipeline:")
    print(f"   python -m src.pipeline --t1 {args.out}/new_construction/sample_001/t1.tif \\")
    print(f"       --t2 {args.out}/new_construction/sample_001/t2.tif \\")
    print(f"       --out outputs/sample_run --dummy")
    print()


if __name__ == "__main__":
    main()
