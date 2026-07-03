"""
Stage 4 — Measurement & rule check.

Tool: shapely + image GSD (ground sample distance) metadata
In:   Polygons + class labels + parcel / red-zone boundary layer
Out:  Area in m², encroachment beyond parcel, setback distance, red-zone overlap
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.stage3_segment_classify import ClassifiedRegion
from src.utils.geo import pixel_polygon_to_world, get_gsd_m, pixel_area_to_m2


@dataclass
class Violation:
    class_name: str
    confidence: float
    polygon_world: "object"       # shapely Polygon in the raster's world CRS
    area_m2: float
    encroaches_parcel: bool
    encroachment_area_m2: float
    setback_violation: bool
    min_setback_m: Optional[float]
    red_zone_overlap: bool
    red_zone_overlap_m2: float


def measure_violations(
    classified_regions: list[ClassifiedRegion],
    transform,
    crs,
    parcel_boundary_path: Optional[str],
    red_zone_boundary_path: Optional[str],
    setback_distance_m: float,
) -> list[Violation]:
    from shapely.geometry import Polygon
    import geopandas as gpd

    parcel_geom = _load_single_geom(parcel_boundary_path, crs) if parcel_boundary_path else None
    red_zone_geom = _load_single_geom(red_zone_boundary_path, crs) if red_zone_boundary_path else None

    gsd_x, gsd_y = get_gsd_m(transform, crs)

    violations = []
    for region in classified_regions:
        world_coords = pixel_polygon_to_world(region.polygon_px, transform)
        if len(world_coords) < 3:
            continue
        poly = Polygon(world_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        # poly.area is already in the raster's projected units (m^2), since
        # Stage 0 requires a projected CRS and the polygon has already been
        # transformed to world coordinates — no extra GSD scaling needed here.
        # (get_gsd_m is still called above as a guard: it raises if the CRS
        # is geographic/degrees, catching a bad Stage 0 run early.)
        area_m2 = poly.area

        encroaches = False
        encroach_area = 0.0
        setback_violation = False
        min_setback = None
        if parcel_geom is not None:
            outside = poly.difference(parcel_geom)
            encroaches = not outside.is_empty
            encroach_area = outside.area if encroaches else 0.0
            min_setback = poly.distance(parcel_geom.boundary)
            setback_violation = min_setback < setback_distance_m

        red_overlap = False
        red_overlap_area = 0.0
        if red_zone_geom is not None:
            inter = poly.intersection(red_zone_geom)
            red_overlap = not inter.is_empty
            red_overlap_area = inter.area if red_overlap else 0.0

        violations.append(
            Violation(
                class_name=region.class_name,
                confidence=region.confidence,
                polygon_world=poly,
                area_m2=area_m2,
                encroaches_parcel=encroaches,
                encroachment_area_m2=encroach_area,
                setback_violation=setback_violation,
                min_setback_m=min_setback,
                red_zone_overlap=red_overlap,
                red_zone_overlap_m2=red_overlap_area,
            )
        )

    return violations


def _load_single_geom(path: str, target_crs):
    import geopandas as gpd
    from shapely.ops import unary_union

    gdf = gpd.read_file(path)
    if gdf.crs is not None and target_crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return unary_union(gdf.geometry)
