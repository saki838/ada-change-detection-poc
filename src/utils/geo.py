"""Geo helpers shared across stages: pixel<->world transforms, GSD lookup."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def pixel_polygon_to_world(pixel_coords: np.ndarray, transform) -> np.ndarray:
    """
    Convert an (N, 2) array of (col, row) pixel coordinates into world
    coordinates using a rasterio Affine transform.
    """
    xs, ys = [], []
    for col, row in pixel_coords:
        x, y = transform * (col, row)
        xs.append(x)
        ys.append(y)
    return np.column_stack([xs, ys])


def get_gsd_m(raster_transform, crs) -> Tuple[float, float]:
    """
    Ground sample distance (meters/pixel) in x and y from a rasterio affine
    transform. Assumes the CRS is projected in meters; if it's geographic
    (degrees), the caller should reproject first (see stage0_align).
    """
    gsd_x = abs(raster_transform.a)
    gsd_y = abs(raster_transform.e)
    if crs is not None and crs.is_geographic:
        raise ValueError(
            "Raster CRS is geographic (degrees), not projected. "
            "Reproject to a metric CRS (e.g. local UTM zone) in Stage 0 "
            "before computing GSD/areas."
        )
    return gsd_x, gsd_y


def pixel_area_to_m2(pixel_area: float, gsd_x_m: float, gsd_y_m: float) -> float:
    return pixel_area * gsd_x_m * gsd_y_m
