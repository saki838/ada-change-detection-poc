"""
Stage 0 — Input prep.

Tool: GDAL / rasterio
In:   Drone RGB or orthomosaic (T2) + approved-plan / baseline raster (T1)
Out:  Two images aligned to the same CRS, same extent and resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedPair:
    t1_rgb: np.ndarray          # (H, W, 3) uint8
    t2_rgb: np.ndarray          # (H, W, 3) uint8
    transform: "object"         # rasterio.Affine, world<->pixel for both arrays
    crs: "object"                # rasterio.crs.CRS


def align_rasters(
    t1_path: str,
    t2_path: str,
    target_resolution_m: float = 0.10,
    resampling_method: str = "bilinear",
) -> AlignedPair:
    """
    Reproject/resample T1 and T2 onto a shared grid: same CRS, same pixel
    size, same extent (intersection of the two footprints), so downstream
    stages can assume pixel (r, c) in T1 == pixel (r, c) in T2 == same
    ground location.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.enums import Resampling as ResamplingEnum

    resampling = getattr(ResamplingEnum, resampling_method)

    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        # Use T1's CRS as the common target CRS (baseline / approved plan is
        # usually the survey-of-record). Change to src2.crs if T2 is trusted more.
        dst_crs = src1.crs

        # Intersection of the two footprints in the target CRS.
        from rasterio.warp import transform_bounds

        b1 = src1.bounds
        b2 = transform_bounds(src2.crs, dst_crs, *src2.bounds)
        left = max(b1.left, b2[0])
        bottom = max(b1.bottom, b2[1])
        right = min(b1.right, b2[2])
        top = min(b1.top, b2[3])
        if left >= right or bottom >= top:
            raise ValueError(
                "T1 and T2 rasters do not overlap after reprojecting to a "
                "common CRS — check the input footprints."
            )

        width = int((right - left) / target_resolution_m)
        height = int((top - bottom) / target_resolution_m)
        from rasterio.transform import from_bounds

        dst_transform = from_bounds(left, bottom, right, top, width, height)

        t1_arr = _reproject_to_grid(src1, dst_crs, dst_transform, width, height, resampling)
        t2_arr = _reproject_to_grid(src2, dst_crs, dst_transform, width, height, resampling)

    return AlignedPair(t1_rgb=t1_arr, t2_rgb=t2_arr, transform=dst_transform, crs=dst_crs)


def _reproject_to_grid(src, dst_crs, dst_transform, width, height, resampling) -> np.ndarray:
    import numpy as np
    from rasterio.warp import reproject

    n_bands = min(src.count, 3)  # RGB only, drop alpha/extra bands if present
    dst = np.zeros((n_bands, height, width), dtype=src.dtypes[0])

    for band in range(1, n_bands + 1):
        reproject(
            source=rasterio_band(src, band),
            destination=dst[band - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    rgb = np.transpose(dst, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = _to_uint8(rgb)
    return rgb


def rasterio_band(src, band_idx):
    return src.read(band_idx)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale non-uint8 imagery (e.g. 16-bit drone orthos) into 0-255."""
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    return scaled.astype(np.uint8)
