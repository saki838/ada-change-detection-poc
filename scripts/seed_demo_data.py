"""Seed demo data for the ADA enforcement dashboard.

Creates 10+ realistic violation cases around Agra with:
  - Various statuses (detected, assigned, field_verified, enforcement_ready, etc.)
  - Various severities (critical, high, medium, low)
  - Various zone types (heritage, riverfront, green_belt, commercial, residential)
  - Evidence records
  - Enforcement notices
  - Zone boundaries
  - Sanctioned plans

Usage:
    python scripts/seed_demo_data.py [--db-path data/ada.db]

After seeding, reload the dashboard at http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy.orm import Session

from src.database.connection import get_engine
from src.database.models import (
    Case,
    CaseEvidence,
    CaseStatus,
    EnforcementNotice,
    EvidenceType,
    NoticeStatus,
    PipelineRun,
    SanctionedPlan,
    Severity,
    ZoneBoundary,
    ZoneType,
    init_db,
)
from src.services.auth_service import AuthService


# ── Helpers ─────────────────────────────────────────────────────────

def _now(days_ago: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def _point_geojson(lat: float, lng: float) -> str:
    return json.dumps({"type": "Point", "coordinates": [lng, lat]})


def _polygon_geojson(coords: list[list[float]]) -> str:
    """coords: list of [lng, lat] pairs forming a closed ring."""
    return json.dumps({"type": "Polygon", "coordinates": [coords]})


# ── Realistic Agra Locations ────────────────────────────────────────
# Coordinates around Agra, Uttar Pradesh, India

LOCATIONS = {
    "taj_mahal":             {"lat": 27.1751, "lng": 78.0421, "zone": "heritage"},
    "taj_east_gate":         {"lat": 27.1730, "lng": 78.0480, "zone": "heritage"},
    "agra_fort":             {"lat": 27.1794, "lng": 78.0213, "zone": "heritage"},
    "yamuna_riverfront_n":   {"lat": 27.1900, "lng": 78.0100, "zone": "riverfront"},
    "yamuna_riverfront_s":   {"lat": 27.1650, "lng": 78.0050, "zone": "riverfront"},
    "mehtab_bagh":           {"lat": 27.1811, "lng": 78.0412, "zone": "green_belt"},
    "sikandra_residential":  {"lat": 27.2190, "lng": 77.9540, "zone": "residential"},
    "sikandra_commercial":   {"lat": 27.2150, "lng": 77.9580, "zone": "commercial"},
    "fatehpur_sikri_road":   {"lat": 27.1000, "lng": 77.8600, "zone": "industrial"},
    "dayalbagh":             {"lat": 27.2150, "lng": 78.0050, "zone": "residential"},
    "kamla_nagar":           {"lat": 27.2000, "lng": 78.0200, "zone": "commercial"},
    "trans_yamuna":          {"lat": 27.1600, "lng": 78.0500, "zone": "residential"},
    "green_belt_north":      {"lat": 27.2400, "lng": 78.0300, "zone": "green_belt"},
    "riverfront_encroach":   {"lat": 27.1700, "lng": 78.0080, "zone": "riverfront"},
}


# ── Demo Violation Cases ────────────────────────────────────────────

DEMO_CASES = [
    {
        "label": "Taj Mahal buffer zone — New construction",
        "loc": "taj_east_gate",
        "class": "new_construction",
        "area": 340.5,
        "confidence": 0.94,
        "severity": Severity.CRITICAL,
        "status": CaseStatus.DETECTED,
        "description": "Unapproved commercial structure detected within 200m of Taj Mahal east gate. Immediate escalation required per heritage zone regulations.",
        "notes": "Flagged by automated patrol on 2026-07-01. Site inspection scheduled.",
    },
    {
        "label": "Yamuna floodplain — Encroachment",
        "loc": "yamuna_riverfront_n",
        "class": "encroachment",
        "area": 890.2,
        "confidence": 0.97,
        "severity": Severity.CRITICAL,
        "status": CaseStatus.ASSIGNED,
        "description": "Encroachment into Yamuna river floodplain — 890m² of unauthorized paving detected. Violates Riverfront Protection Zone (No Construction Zone).",
        "assigned_to": "Amit Sharma",
        "notes": "Assigned to Officer Sharma for field verification. Previous notice served in 2024.",
    },
    {
        "label": "Sikandra commercial — Horizontal expansion",
        "loc": "sikandra_commercial",
        "class": "horizontal_expansion",
        "area": 125.0,
        "confidence": 0.88,
        "severity": Severity.HIGH,
        "status": CaseStatus.FIELD_VERIFIED,
        "description": "Unauthorized horizontal expansion of shop front by 125m² into pedestrian ROW. Violation of commercial zone setback requirements.",
        "assigned_to": "Priya Singh",
        "notes": "Field visit completed 2026-07-02. Photos attached. Expansion confirmed beyond sanctioned plan.",
    },
    {
        "label": "Mehtab Bagh green belt — Unauthorized construction",
        "loc": "mehtab_bagh",
        "class": "new_construction",
        "area": 210.0,
        "confidence": 0.92,
        "severity": Severity.CRITICAL,
        "status": CaseStatus.ENFORCEMENT_READY,
        "description": "Unauthorized structure in Mehtab Bagh green belt zone. 210m² building detected — no sanctioned plan on record. Green belt is a no-construction zone.",
        "assigned_to": "Amit Sharma",
        "notes": "Permit reconciliation complete. No matching plan found. Enforcement notice prepared.",
    },
    {
        "label": "Fatehpur Sikri road — Industrial construction",
        "loc": "fatehpur_sikri_road",
        "class": "vertical_expansion",
        "area": 450.0,
        "confidence": 0.85,
        "severity": Severity.HIGH,
        "status": CaseStatus.DETECTED,
        "description": "Vertical expansion detected in industrial zone — 3 additional floors beyond approved height. Potential safety hazard.",
        "notes": "",
    },
    {
        "label": "Agra Fort buffer — Vegetation clearance",
        "loc": "agra_fort",
        "class": "vegetation_clearance",
        "area": 5600.0,
        "confidence": 0.91,
        "severity": Severity.HIGH,
        "status": CaseStatus.ASSIGNED,
        "description": "Large-scale vegetation clearance (5,600m²) detected in Agra Fort heritage buffer zone. Environmental impact assessment required.",
        "assigned_to": "Priya Singh",
        "notes": "Heritage committee notified. Awaiting field report.",
    },
    {
        "label": "Kamla Nagar — Rooftop extension",
        "loc": "kamla_nagar",
        "class": "vertical_expansion",
        "area": 85.0,
        "confidence": 0.78,
        "severity": Severity.MEDIUM,
        "status": CaseStatus.RESOLVED,
        "description": "Unauthorized rooftop extension in commercial zone. 85m² added without permit. Resolved — structure regularized with penalty.",
        "assigned_to": "Amit Sharma",
        "notes": "Penalty of ₹2,40,000 collected. Structure regularized under Section 345(2) of U.P. Municipal Act.",
    },
    {
        "label": "Trans Yamuna — Unauthorized paving",
        "loc": "trans_yamuna",
        "class": "unauthorized_paving",
        "area": 320.0,
        "confidence": 0.82,
        "severity": Severity.MEDIUM,
        "status": CaseStatus.NOTICE_ISSUED,
        "description": "Unauthorized paving of residential plot for commercial parking. 320m² paved area without environmental clearance.",
        "assigned_to": "Priya Singh",
        "notes": "Show cause notice issued 2026-07-03. 15-day response period.",
    },
    {
        "label": "Green belt north — Encroachment",
        "loc": "green_belt_north",
        "class": "encroachment",
        "area": 1500.0,
        "confidence": 0.95,
        "severity": Severity.CRITICAL,
        "status": CaseStatus.ESCALATED,
        "description": "Large-scale encroachment into protected green belt — 1,500m². Referred to High Court monitoring committee.",
        "assigned_to": "Rajesh Kumar",
        "notes": "Escalated to Zonal Supervisor. Matter referred to Hon'ble High Court monitoring committee for green belt protection.",
    },
    {
        "label": "Riverfront south — New construction",
        "loc": "riverfront_encroach",
        "class": "new_construction",
        "area": 180.0,
        "confidence": 0.89,
        "severity": Severity.HIGH,
        "status": CaseStatus.DETECTED,
        "description": "New construction detected in Yamuna riverfront restricted zone. 180m² structure within 50m of high-tide line.",
        "notes": "Riverfront Protection Act applicable. Immediate notice recommended.",
    },
    {
        "label": "Dayalbagh — Home extension",
        "loc": "dayalbagh",
        "class": "horizontal_expansion",
        "area": 45.0,
        "confidence": 0.72,
        "severity": Severity.LOW,
        "status": CaseStatus.RESOLVED,
        "description": "Minor horizontal extension of residential property — 45m². Within permissible limits. Compound wall constructed without prior approval.",
        "assigned_to": "Priya Singh",
        "notes": "Regularized with nominal fee of ₹15,000. No structural violation.",
    },
    {
        "label": "Sikandra residential — Setback violation",
        "loc": "sikandra_residential",
        "class": "horizontal_expansion",
        "area": 30.0,
        "confidence": 0.65,
        "severity": Severity.LOW,
        "status": CaseStatus.FIELD_VERIFIED,
        "description": "Minor setback violation in residential zone — 30m² balcony extension beyond permitted building line.",
        "assigned_to": "Amit Sharma",
        "notes": "Field verification confirmed 0.8m setback violation. Within negotiable range.",
    },
]


# ── Zone Boundaries (Agra) ──────────────────────────────────────────

ZONE_BOUNDARIES = [
    {
        "zone_type": ZoneType.HERITAGE,
        "name": "Taj Mahal Protected Zone (200m buffer)",
        "coords": [
            [78.038, 27.172], [78.046, 27.172],
            [78.046, 27.178], [78.038, 27.178],
            [78.038, 27.172],
        ],
        "setback": 50.0,
        "severity_boost": 2.5,
        "description": "UNESCO World Heritage Site — Absolute construction prohibition within 200m",
    },
    {
        "zone_type": ZoneType.HERITAGE,
        "name": "Agra Fort Buffer Zone",
        "coords": [
            [78.017, 27.176], [78.025, 27.176],
            [78.025, 27.183], [78.017, 27.183],
            [78.017, 27.176],
        ],
        "setback": 30.0,
        "severity_boost": 2.0,
        "description": "ASI Protected Monument — Restricted construction zone",
    },
    {
        "zone_type": ZoneType.RIVERFRONT,
        "name": "Yamuna Riverfront Protection Zone",
        "coords": [
            [78.000, 27.158], [78.055, 27.158],
            [78.055, 27.192], [78.000, 27.192],
            [78.000, 27.158],
        ],
        "setback": 50.0,
        "severity_boost": 2.2,
        "description": "Yamuna Riverfront — No construction within 50m of high-tide line",
    },
    {
        "zone_type": ZoneType.GREEN_BELT,
        "name": "Mehtab Bagh Green Belt",
        "coords": [
            [78.038, 27.179], [78.044, 27.179],
            [78.044, 27.184], [78.038, 27.184],
            [78.038, 27.179],
        ],
        "setback": 15.0,
        "severity_boost": 1.8,
        "description": "Protected green zone — Environmental buffer for Taj Mahal complex",
    },
]


# ── Sanctioned Plans ────────────────────────────────────────────────

SANCTIONED_PLANS = [
    {
        "parcel_id": "AGR-SIK-001",
        "plan_number": "ADA/SIK/2025/0042",
        "coords": [
            [77.956, 27.216], [77.960, 27.216],
            [77.960, 27.220], [77.956, 27.220],
            [77.956, 27.216],
        ],
        "area": 150.0,
        "height": 12.0,
        "floors": 3,
        "approval_days_ago": 180,
        "expiry_days_ago": None,
    },
    {
        "parcel_id": "AGR-DAY-002",
        "plan_number": "ADA/DAY/2026/0017",
        "coords": [
            [78.003, 27.213], [78.006, 27.213],
            [78.006, 27.216], [78.003, 27.216],
            [78.003, 27.213],
        ],
        "area": 120.0,
        "height": 9.0,
        "floors": 2,
        "approval_days_ago": 90,
        "expiry_days_ago": None,
    },
]


# ── Main Seed Function ──────────────────────────────────────────────

def seed_demo_data(db_path: str) -> dict:
    """Seed demo data into the database. Returns summary counts."""
    init_db(db_path)
    engine = get_engine(db_path)
    counts = {"cases": 0, "evidence": 0, "notices": 0, "zones": 0, "plans": 0}

    with Session(engine) as session:
        # ── 0. Clear existing demo data (idempotent re-runs) ──
        existing = session.query(Case).filter(Case.case_number.like("ADA-DEMO-%")).count()
        if existing > 0:
            print(f"   Clearing {existing} existing demo cases...")
            for table in [EnforcementNotice, CaseEvidence, Case, PipelineRun, SanctionedPlan, ZoneBoundary]:
                session.query(table).delete()
            session.commit()

        # ── 1. Create a PipelineRun ──
        run_id = uuid.uuid4().hex[:8]
        run = PipelineRun(
            id=run_id,
            t1_filename="demo/agra_t1_2025.tif",
            t2_filename="demo/agra_t2_2026.tif",
            parcel_filename="demo/parcel.geojson",
            status="completed",
            violation_count=len(DEMO_CASES),
            created_at=_now(days_ago=7),
            completed_at=_now(days_ago=6),
        )
        session.add(run)
        session.flush()

        # ── 2. Create Cases ──
        for i, cd in enumerate(DEMO_CASES):
            loc = LOCATIONS[cd["loc"]]
            created_at = _now(days_ago=7 - i)  # stagger creation dates
            case_number = f"ADA-DEMO-{i+1:04d}"

            case = Case(
                case_number=case_number,
                run_id=run_id,
                violation_class=cd["class"],
                confidence=cd["confidence"],
                area_m2=cd["area"],
                location_geojson=_point_geojson(loc["lat"], loc["lng"]),
                zone_type=loc["zone"],
                severity=cd["severity"],
                status=cd["status"],
                assigned_to=cd.get("assigned_to"),
                description=cd["description"],
                notes=cd.get("notes", ""),
                created_at=created_at,
                updated_at=created_at,
            )
            session.add(case)
            session.flush()
            counts["cases"] += 1

            # ── 3. Add Evidence for cases past DETECTED ──
            if cd["status"] in (
                CaseStatus.FIELD_VERIFIED, CaseStatus.ENFORCEMENT_READY,
                CaseStatus.NOTICE_ISSUED, CaseStatus.RESOLVED,
                CaseStatus.ESCALATED,
            ):
                evidence = CaseEvidence(
                    case_id=case.id,
                    evidence_type=EvidenceType.DRONE_IMAGE,
                    file_path=f"outputs/evidence/{case.id}/drone_{cd['loc']}.jpg",
                    description=f"Drone imagery — {cd['label']}",
                    uploaded_at=created_at + timedelta(hours=2),
                    uploaded_by=cd.get("assigned_to", "system"),
                )
                session.add(evidence)
                counts["evidence"] += 1

            # ── 4. Add Enforcement Notice for NOTICE_ISSUED+ cases ──
            if cd["status"] in (
                CaseStatus.NOTICE_ISSUED, CaseStatus.RESOLVED,
                CaseStatus.ESCALATED,
            ):
                notice_no = f"ADA/NOTICE/2026/{i+1:04d}"
                notice = EnforcementNotice(
                    case_id=case.id,
                    notice_number=notice_no,
                    status=NoticeStatus.ISSUED,
                    file_path=f"outputs/notices/{notice_no}.pdf",
                    legal_reference="U.P. Urban Planning & Development Act, 1973 Section 35(2); "
                                    "Model Building Bye-Laws 2016 Chapter VIII",
                    generated_at=created_at + timedelta(days=2),
                    issued_at=created_at + timedelta(days=3),
                )
                session.add(notice)
                counts["notices"] += 1

        # ── 5. Create Zone Boundaries ──
        for zb in ZONE_BOUNDARIES:
            poly = _polygon_geojson(zb["coords"])
            boundary = ZoneBoundary(
                zone_type=zb["zone_type"],
                name=zb["name"],
                boundary_geojson=poly,
                setback_distance_m=zb["setback"],
                severity_boost=zb["severity_boost"],
                description=zb["description"],
                created_at=_now(days_ago=30),
            )
            session.add(boundary)
            counts["zones"] += 1

        # ── 6. Create Sanctioned Plans ──
        for sp in SANCTIONED_PLANS:
            poly = _polygon_geojson(sp["coords"])
            plan = SanctionedPlan(
                parcel_id=sp["parcel_id"],
                plan_number=sp["plan_number"],
                approved_footprint_geojson=poly,
                approved_area_m2=sp["area"],
                approved_height_m=sp["height"],
                approved_floors=sp["floors"],
                approval_date=_now(days_ago=sp["approval_days_ago"]),
                expiry_date=_now(days_ago=sp["expiry_days_ago"]) if sp["expiry_days_ago"] else None,
                status="active",
                created_at=_now(days_ago=sp["approval_days_ago"]),
            )
            session.add(plan)
            counts["plans"] += 1

        session.commit()

    return counts


def main():
    parser = argparse.ArgumentParser(description="Seed demo data for ADA dashboard")
    parser.add_argument(
        "--db-path",
        default="data/ada.db",
        help="Path to SQLite database (default: data/ada.db)",
    )
    args = parser.parse_args()

    print("🏗️  Seeding demo data...")
    print(f"   Database: {args.db_path}")
    print()

    counts = seed_demo_data(args.db_path)

    print(f"   ✅ {counts['cases']} violation cases created")
    print(f"   ✅ {counts['evidence']} evidence records created")
    print(f"   ✅ {counts['notices']} enforcement notices created")
    print(f"   ✅ {counts['zones']} zone boundaries created")
    print(f"   ✅ {counts['plans']} sanctioned plans created")
    print()
    print("🎉 Demo data seeded successfully!")
    print()
    print("Reload http://127.0.0.1:8000 and log in with:")
    print("   admin@ada.gov.in / admin123")
    print()
    print("You should see the dashboard populated with cases,")
    print("map markers, stats, and analytics data.")


if __name__ == "__main__":
    main()
