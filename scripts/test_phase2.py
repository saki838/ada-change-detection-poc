"""Phase 2 smoke test — verifies Stage 4.5 permit reconciliation, case service,
and notice PDF generator.

Run this after:
    pip install sqlalchemy>=2.0 fpdf2

Usage:
    python scripts/test_phase2.py

It will:
  1. Create an in-memory database with test sanctioned plans
  2. Create mock violations (simulating Stage 4 output)
  3. Run Stage 4.5 permit reconciliation
  4. Create cases from reconciled violations
  5. Generate an enforcement notice PDF
  6. Verify all outputs
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def make_mock_violation(class_name, area_m2, x, y):
    """Create a mock Violation-like dict for testing."""
    from shapely.geometry import Polygon

    # Create a simple square polygon in world coords
    half = (area_m2 ** 0.5) / 2
    poly = Polygon([
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
    ])

    return {
        "violation": type("MockViolation", (), {
            "class_name": class_name,
            "confidence": 0.92,
            "area_m2": area_m2,
            "polygon_world": poly,
            "encroaches_parcel": True,
            "encroachment_area_m2": area_m2 * 0.1,
            "setback_violation": True,
            "min_setback_m": 1.5,
            "red_zone_overlap": False,
            "red_zone_overlap_m2": 0.0,
        })(),
        "permit_reconciliation": None,  # will be filled by Stage 4.5
    }


def test_stage4_5_permit_reconciliation():
    """Test that Stage 4.5 correctly matches violations to sanctioned plans."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.database.models import Base, SanctionedPlan
    from src.stage4_5_permit_reconciliation import reconcile_violations

    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Insert a sanctioned plan
        plan_poly = {"type": "Polygon", "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]}
        plan = SanctionedPlan(
            parcel_id="PARCEL-001",
            plan_number="ADA-BP-2025-0042",
            approved_footprint_geojson=json.dumps(plan_poly),
            approved_area_m2=1000.0,
            approved_floors=2,
            approval_date="2025-01-15",
            status="active",
        )
        session.add(plan)
        session.commit()

        # Create mock violations
        mock_violations = [
            make_mock_violation("new_construction", 120.0, 50, 50),     # within plan, slightly over area
            make_mock_violation("encroachment", 2000.0, 200, 200),      # far away — no permit
            make_mock_violation("horizontal_expansion", 800.0, 50, 50), # within plan area
        ]

        # Extract just the Violation objects
        violations = [m["violation"] for m in mock_violations]

        # Run Stage 4.5
        results = reconcile_violations(violations, session, "EPSG:32643")

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Violation 1: within plan area, close to plan polygon
        assert results[0]["permit_reconciliation"].permit_found, "Violation 1 should match a plan"
        assert results[0]["permit_reconciliation"].plan_number == "ADA-BP-2025-0042"
        print(f"  ✅ Violation 1: permit={results[0]['permit_reconciliation'].status}")

        # Violation 2: far away — no permit
        assert not results[1]["permit_reconciliation"].permit_found, "Violation 2 should have no plan"
        print(f"  ✅ Violation 2: permit={results[1]['permit_reconciliation'].status}")

        # Violation 3: within plan area
        assert results[2]["permit_reconciliation"].permit_found, "Violation 3 should match a plan"
        print(f"  ✅ Violation 3: permit={results[2]['permit_reconciliation'].status}")

    engine.dispose()
    print("  ✅ Stage 4.5 Permit Reconciliation: all assertions passed")


def test_case_service():
    """Test the Case Service — creation, queries, status updates."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.database.models import Base, PipelineRun
    from src.services.case_service import CaseService

    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Create a pipeline run
        run = PipelineRun(
            id="test_pipeline_001",
            t1_filename="baseline.tif",
            t2_filename="drone.tif",
            status="completed",
        )
        session.add(run)
        session.commit()

        # Create mock reconciled violations
        mock_data = [
            make_mock_violation("new_construction", 150.0, 50, 50),
            make_mock_violation("encroachment", 25.0, 10, 10),
        ]

        # Set permit reconciliation results
        from src.stage4_5_permit_reconciliation import PermitReconciliation
        for i, m in enumerate(mock_data):
            if i == 0:
                m["permit_reconciliation"] = PermitReconciliation(
                    permit_found=True, plan_number="BP-001",
                    approved_area_m2=100.0, excess_area_m2=50.0,
                    status="excess_area",
                    summary="Exceeds approved area by 50.0m²",
                )
            else:
                m["permit_reconciliation"] = PermitReconciliation(
                    permit_found=False, status="no_permit_found",
                    summary="No permit found",
                )

        # Create cases
        service = CaseService()
        cases = service.create_cases_from_violations(
            session=session,
            run_id=run.id,
            reconciled_violations=mock_data,
            zone_type="heritage",
        )

        assert len(cases) == 2, f"Expected 2 cases, got {len(cases)}"
        assert cases[0].case_number.startswith("ADA-")
        assert cases[0].severity == "critical", f"Expected critical, got {cases[0].severity}"
        print(f"  ✅ Created 2 cases: {cases[0].case_number} ({cases[0].severity}), "
              f"{cases[1].case_number} ({cases[1].severity})")

        # Query cases
        all_cases = service.list_cases(session)
        assert len(all_cases) == 2

        # Update status
        updated = service.update_status(
            session, cases[0].id, "assigned",
            notes="Assigned for field verification",
        )
        assert updated is not None
        assert updated.status == "assigned"
        print(f"  ✅ Case {cases[0].case_number} status updated to 'assigned'")

        # Assign officer
        assigned = service.assign_case(session, cases[0].id, "Officer Sharma")
        assert assigned.assigned_to == "Officer Sharma"
        print(f"  ✅ Case assigned to Officer Sharma")

        # Get stats
        stats = service.get_case_stats(session)
        assert stats["total"] == 2
        print(f"  ✅ Case stats: {stats['total']} total, "
              f"{stats['by_severity']['critical']} critical")

    engine.dispose()
    print("  ✅ Case Service: all assertions passed")


def test_notice_generator():
    """Test that the notice PDF generator creates a valid PDF file."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.database.models import Base, PipelineRun, Case
    from src.services.notice_generator import generate_notice

    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        run = PipelineRun(id="notice_test", t1_filename="a.tif", t2_filename="b.tif")
        case = Case(
            case_number="ADA-2026-9999",
            run_id=run.id,
            violation_class="new_construction",
            confidence=0.95,
            area_m2=250.0,
            zone_type="heritage",
            severity="critical",
            description="Unauthorized construction in heritage zone",
        )
        session.add_all([run, case])
        session.commit()
        session.refresh(case)

        # Generate PDF
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = generate_notice(case, tmp)
            assert pathlib.Path(pdf_path).exists()
            assert pathlib.Path(pdf_path).stat().st_size > 1000  # at least 1KB
            print(f"  ✅ Notice PDF generated: {pathlib.Path(pdf_path).name} "
                  f"({pathlib.Path(pdf_path).stat().st_size:,} bytes)")

    engine.dispose()
    print("  ✅ Notice Generator: all assertions passed")


def main():
    print("🔍 Phase 2 Smoke Test\n")

    print("  Testing Stage 4.5 — Permit Reconciliation...")
    test_stage4_5_permit_reconciliation()

    print("\n  Testing Case Service...")
    test_case_service()

    print("\n  Testing Notice Generator...")
    test_notice_generator()

    print("\n" + "=" * 50)
    print("🎉 Phase 2: all tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
