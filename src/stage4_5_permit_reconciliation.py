"""
Stage 4.5 — Permit Reconciliation.

Compares detected violations from Stage 4 against sanctioned building plans
stored in the database. Flags:

  - Excess area (detected footprint > sanctioned area)
  - Unapproved extension (polygon extends beyond sanctioned footprint)
  - Construction without permit (no matching sanctioned plan found)

In/Out:
    In:  list[Violation] from Stage 4 + database session
    Out: list[dict] — each violation enriched with permit reconciliation data

This stage runs AFTER Stage 4 measurement and BEFORE case creation, so that
the Case Service can incorporate permit status into severity scoring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import shape, Polygon
from sqlalchemy.orm import Session

from src.database.models import SanctionedPlan
from src.stage4_measurement import Violation


@dataclass
class PermitReconciliation:
    """Result of comparing a detected violation against sanctioned plans."""

    # Whether a matching sanctioned plan was found
    permit_found: bool = False

    # The matched plan details (if found)
    plan_number: Optional[str] = None
    parcel_id: Optional[str] = None
    approved_area_m2: Optional[float] = None
    approved_floors: Optional[int] = None

    # Area comparison (if plan found)
    excess_area_m2: float = 0.0
    excess_area_pct: float = 0.0  # percentage over approved
    area_compliant: bool = True

    # Footprint comparison (if plan found)
    footprint_compliant: bool = True
    unapproved_extension_m2: float = 0.0

    # Overall status
    status: str = "no_permit_found"  # compliant | excess_area | unapproved_extension | no_permit_found

    # Human-readable summary
    summary: str = ""


def reconcile_violations(
    violations: list[Violation],
    db_session: Session,
    crs=None,
) -> list[dict]:
    """Enrich each violation with permit reconciliation data.

    For each violation:
      1. Determine which parcel it falls within (point-in-polygon against
         all sanctioned plans).
      2. If a matching plan is found, compare detected vs sanctioned.
      3. Return structured reconciliation result.

    Args:
        violations: List of Violation dataclass instances from Stage 4.
        db_session: SQLAlchemy session for querying sanctioned_plans.
        crs: The CRS of the violations (reserved for future spatial
             reprojection validation; not yet used).

    Returns:
        List of dicts, one per violation, with original violation data
        plus permit_reconciliation key.
    """
    # Load all active sanctioned plans from the database
    plans = db_session.query(SanctionedPlan).filter_by(status="active").all()

    # Build spatial index: approved footprint polygons
    plan_polygons = []
    for plan in plans:
        try:
            geom = json.loads(plan.approved_footprint_geojson)
            poly = shape(geom)
            if not poly.is_valid:
                poly = poly.buffer(0)
            plan_polygons.append((plan, poly))
        except (json.JSONDecodeError, Exception):
            continue  # skip malformed plans

    results = []
    for violation in violations:
        poly = violation.polygon_world
        if poly is None or poly.is_empty:
            results.append(_no_plan_result(violation, "No valid polygon"))
            continue

        # Find matching plan by spatial containment/intersection
        matching_plan = None
        matching_poly = None
        for plan, plan_poly in plan_polygons:
            if poly.intersects(plan_poly) or poly.within(plan_poly.buffer(5)):
                matching_plan = plan
                matching_poly = plan_poly
                break

        if matching_plan is None:
            # No sanctioned plan found — flag as "construction without permit"
            results.append(_no_plan_result(
                violation,
                "No matching sanctioned plan found for this location.",
            ))
            continue

        # Plan found — compare areas and footprints
        reconciliation = _compare_with_plan(
            violation, poly, matching_plan, matching_poly,
        )
        results.append({
            "violation": violation,
            "permit_reconciliation": reconciliation,
        })

    return results


def _compare_with_plan(
    violation: Violation,
    detected_poly: Polygon,
    plan: SanctionedPlan,
    plan_poly: Polygon,
) -> PermitReconciliation:
    """Compare a detected violation against a sanctioned plan."""
    # Area comparison
    excess = max(0.0, violation.area_m2 - plan.approved_area_m2)
    excess_pct = (
        (excess / plan.approved_area_m2 * 100)
        if plan.approved_area_m2 > 0
        else 0.0
    )
    area_compliant = excess <= 1.0  # allow 1 m² tolerance

    # Footprint comparison: does detected extend beyond approved?
    outside = detected_poly.difference(plan_poly)
    unapproved_ext = outside.area if not outside.is_empty else 0.0
    footprint_compliant = unapproved_ext <= 1.0  # allow 1 m² tolerance

    # Determine overall status
    if not area_compliant and not footprint_compliant:
        status = "unapproved_extension"
        summary = (
            f"Exceeds approved area by {excess:.1f}m² ({excess_pct:.0f}%) "
            f"and extends beyond approved footprint by {unapproved_ext:.1f}m²"
        )
    elif not area_compliant:
        status = "excess_area"
        summary = f"Exceeds approved area by {excess:.1f}m² ({excess_pct:.0f}%)"
    elif not footprint_compliant:
        status = "unapproved_extension"
        summary = f"Extends beyond approved footprint by {unapproved_ext:.1f}m²"
    else:
        status = "compliant"
        summary = f"Within approved plan (area: {violation.area_m2:.1f}m² vs {plan.approved_area_m2:.1f}m²)"

    return PermitReconciliation(
        permit_found=True,
        plan_number=plan.plan_number,
        parcel_id=plan.parcel_id,
        approved_area_m2=plan.approved_area_m2,
        approved_floors=plan.approved_floors,
        excess_area_m2=round(excess, 2),
        excess_area_pct=round(excess_pct, 1),
        area_compliant=area_compliant,
        footprint_compliant=footprint_compliant,
        unapproved_extension_m2=round(unapproved_ext, 2),
        status=status,
        summary=summary,
    )


def _no_plan_result(
    violation: Violation,
    reason: str = "",
) -> dict:
    """Create a result entry when no matching plan is found."""
    return {
        "violation": violation,
        "permit_reconciliation": PermitReconciliation(
            permit_found=False,
            status="no_permit_found",
            summary=reason or "No matching sanctioned plan found.",
        ),
    }
