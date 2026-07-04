"""Case Management Service — violation detection → tracked case lifecycle.

Responsibilities:
  1. Create Cases in the database from pipeline violations + permit reconciliation
  2. Compute severity using ZoneRulesEngine (zone type + violation class + area)
  3. Generate human-readable case numbers (ADA-2026-XXXX)
  4. Assign/unassign cases to enforcement officers
  5. Update case status through its lifecycle
  6. Query/search/filter cases from the database

Usage:
    from src.services.case_service import CaseService
    from src.database.connection import get_engine
    from sqlalchemy.orm import Session

    service = CaseService()
    with Session(get_engine()) as session:
        cases = service.create_cases_from_violations(
            session=session,
            run_id="abc123",
            reconciled_violations=[...],  # from Stage 4.5
            zone_type="heritage",
        )
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.database.models import (
    Case,
    CaseEvidence,
    CaseStatus,
    EnforcementNotice,
    NoticeStatus,
    PipelineRun,
    SanctionedPlan,
)
from src.services.zone_rules import ZoneRulesEngine


class CaseService:
    """Service layer for violation case lifecycle management."""

    def __init__(self, rules_engine: Optional[ZoneRulesEngine] = None):
        self._rules = rules_engine or ZoneRulesEngine()

    # ── Case Creation ───────────────────────────────────────────

    def create_cases_from_violations(
        self,
        session: Session,
        run_id: str,
        reconciled_violations: list[dict],
        zone_type: Optional[str] = None,
        default_zone: str = "other",
    ) -> list[Case]:
        """Create Case records from reconciled pipeline violations.

        Args:
            session: Active SQLAlchemy session.
            run_id: The pipeline run ID these violations belong to.
            reconciled_violations: Output from Stage 4.5 — list of dicts
                with 'violation' and 'permit_reconciliation' keys.
            zone_type: Override zone type for all violations. If None,
                determined from spatial lookup (future).
            default_zone: Fallback zone type when zone_type is None.

        Returns:
            List of created Case ORM objects (already committed).
        """
        created_cases = []
        next_number = self._next_case_number(session)

        for i, item in enumerate(reconciled_violations):
            violation = item["violation"]
            permit = item["permit_reconciliation"]

            # Determine zone type
            zt = zone_type or default_zone

            # Compute severity
            severity = self._compute_case_severity(
                zone_type=zt,
                violation_class=violation.class_name,
                area_m2=violation.area_m2,
                permit_status=permit.status,
            )

            # Build location GeoJSON from the violation polygon
            location_geojson = self._polygon_to_geojson(violation.polygon_world)

            # Build description incorporating permit data
            description = self._build_description(violation, permit)

            case_number = f"ADA-{datetime.now(timezone.utc).year}-{next_number + i:04d}"

            case = Case(
                case_number=case_number,
                run_id=run_id,
                violation_class=violation.class_name,
                confidence=violation.confidence,
                area_m2=violation.area_m2,
                location_geojson=location_geojson,
                zone_type=zt,
                severity=severity,
                status=CaseStatus.DETECTED,
                description=description,
            )
            session.add(case)
            created_cases.append(case)

        session.commit()

        # Refresh to get generated IDs and timestamps
        for c in created_cases:
            session.refresh(c)

        return created_cases

    # ── Case Queries ────────────────────────────────────────────

    def get_case(self, session: Session, case_id: int) -> Optional[Case]:
        """Fetch a single case by ID."""
        return session.query(Case).filter_by(id=case_id).first()

    def get_case_by_number(self, session: Session, case_number: str) -> Optional[Case]:
        """Fetch a single case by its human-readable number."""
        return session.query(Case).filter_by(case_number=case_number).first()

    def list_cases(
        self,
        session: Session,
        status: Optional[str] = None,
        zone_type: Optional[str] = None,
        severity: Optional[str] = None,
        assigned_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Case]:
        """List cases with optional filters. Returns most recent first."""
        query = session.query(Case)

        if status:
            query = query.filter(Case.status == status)
        if zone_type:
            query = query.filter(Case.zone_type == zone_type)
        if severity:
            query = query.filter(Case.severity == severity)
        if assigned_to:
            query = query.filter(Case.assigned_to == assigned_to)

        return (
            query.order_by(Case.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    def count_cases(
        self,
        session: Session,
        status: Optional[str] = None,
        zone_type: Optional[str] = None,
        severity: Optional[str] = None,
        assigned_to: Optional[str] = None,
    ) -> int:
        """Count cases matching filters."""
        query = session.query(Case)
        if status:
            query = query.filter(Case.status == status)
        if zone_type:
            query = query.filter(Case.zone_type == zone_type)
        if severity:
            query = query.filter(Case.severity == severity)
        if assigned_to:
            query = query.filter(Case.assigned_to == assigned_to)
        return query.count()

    # ── Case Workflow ───────────────────────────────────────────

    def update_status(
        self,
        session: Session,
        case_id: int,
        new_status: str,
        notes: Optional[str] = None,
    ) -> Optional[Case]:
        """Update a case's workflow status.

        Valid transitions:
          detected → assigned → field_verified → enforcement_ready
          → notice_issued → resolved | escalated
        """
        case = self.get_case(session, case_id)
        if case is None:
            return None

        case.status = new_status
        if notes:
            existing = case.notes or ""
            case.notes = f"{existing}\n[{datetime.now(timezone.utc).isoformat()}] {notes}".strip()

        session.commit()
        session.refresh(case)
        return case

    def assign_case(
        self,
        session: Session,
        case_id: int,
        officer_name: str,
    ) -> Optional[Case]:
        """Assign a case to an enforcement officer."""
        case = self.get_case(session, case_id)
        if case is None:
            return None

        case.assigned_to = officer_name
        if case.status == CaseStatus.DETECTED:
            case.status = CaseStatus.ASSIGNED

        session.commit()
        session.refresh(case)
        return case

    def add_evidence(
        self,
        session: Session,
        case_id: int,
        evidence_type: str,
        file_path: str,
        description: Optional[str] = None,
        uploaded_by: Optional[str] = None,
    ) -> Optional[CaseEvidence]:
        """Attach evidence (photo, document, etc.) to a case."""
        case = self.get_case(session, case_id)
        if case is None:
            return None

        evidence = CaseEvidence(
            case_id=case.id,
            evidence_type=evidence_type,
            file_path=file_path,
            description=description,
            uploaded_by=uploaded_by,
        )
        session.add(evidence)
        session.commit()
        session.refresh(evidence)
        return evidence

    # ── Case Analytics ──────────────────────────────────────────

    def get_case_stats(self, session: Session) -> dict:
        """Return aggregate case statistics for dashboards."""
        total = self.count_cases(session)
        return {
            "total": total,
            "by_status": {
                s: self.count_cases(session, status=s)
                for s in CaseStatus.CHOICES
            },
            "by_severity": {
                s: self.count_cases(session, severity=s)
                for s in ["low", "medium", "high", "critical"]
            },
            "by_zone": {
                z: self.count_cases(session, zone_type=z)
                for z in ["residential", "commercial", "heritage",
                          "green_belt", "riverfront", "industrial", "other"]
            },
        }

    # ── Private Helpers ─────────────────────────────────────────

    def _compute_case_severity(
        self,
        zone_type: str,
        violation_class: str,
        area_m2: float,
        permit_status: str,
    ) -> str:
        """Compute severity, with a boost for unpermitted construction."""
        base_severity = self._rules.compute_severity(
            zone_type=zone_type,
            violation_class=violation_class,
            area_m2=area_m2,
        )

        # Check for zone-based restriction overrides
        restriction = self._rules.check_zone_restriction(zone_type, violation_class)
        if restriction["severity_override"]:
            return restriction["severity_override"]

        # Boost severity for violations without a permit
        if permit_status == "no_permit_found":
            sev_order = ["low", "medium", "high", "critical"]
            idx = sev_order.index(base_severity)
            return sev_order[min(idx + 1, len(sev_order) - 1)]

        return base_severity

    def _next_case_number(self, session: Session) -> int:
        """Determine the next sequential case number suffix."""
        latest = (
            session.query(Case)
            .order_by(Case.id.desc())
            .first()
        )
        if latest is None:
            return 1
        return latest.id + 1

    def _polygon_to_geojson(self, polygon) -> str:
        """Convert a shapely polygon to a GeoJSON string."""
        try:
            from shapely.geometry import mapping
            return json.dumps(mapping(polygon))
        except Exception:
            return json.dumps({"type": "Point", "coordinates": [0, 0]})

    def _build_description(self, violation, permit) -> str:
        """Build a human-readable violation description."""
        parts = [
            f"Detected: {violation.class_name.replace('_', ' ').title()}",
            f"Area: {violation.area_m2:.1f}m²",
            f"Confidence: {violation.confidence:.0%}",
        ]
        if violation.encroaches_parcel:
            parts.append(f"Encroaches parcel by {violation.encroachment_area_m2:.1f}m²")
        if violation.setback_violation:
            parts.append(f"Setback violation (min: {violation.min_setback_m:.1f}m)")
        if violation.red_zone_overlap:
            parts.append(f"Overlaps restricted zone by {violation.red_zone_overlap_m2:.1f}m²")

        # Add permit info
        parts.append(f"Permit: {permit.summary}")

        return " | ".join(parts)
