"""Phase 1 smoke test — verifies database models, connection, and zone rules.

Run this after installing SQLAlchemy (pip install sqlalchemy>=2.0):

    python scripts/test_phase1.py

It will:
  1. Create all database tables in a temporary in-memory SQLite database
  2. Insert a sample pipeline run and case
  3. Query the case back and verify relationships
  4. Test the zone rules engine severity computation
  5. Clean up (temp db is discarded on exit)
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_database_models():
    """Create tables, insert a pipeline run + case, query back."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from src.database.models import Base, PipelineRun, Case

    # Use an in-memory SQLite database for testing
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Insert a pipeline run
        run = PipelineRun(
            id="test_run_001",
            t1_filename="baseline.tif",
            t2_filename="drone_current.tif",
            parcel_filename="parcel.geojson",
            status="completed",
            violation_count=2,
        )
        session.add(run)
        session.flush()

        # Insert a case linked to the run
        case = Case(
            case_number="ADA-2026-0001",
            run_id=run.id,
            violation_class="new_construction",
            confidence=0.92,
            area_m2=120.5,
            zone_type="heritage",
            severity="critical",
            status="detected",
            description="Unauthorized new structure detected in heritage zone",
        )
        session.add(case)
        session.commit()

        # Query back and validate
        saved_run = session.query(PipelineRun).filter_by(id="test_run_001").first()
        assert saved_run is not None, "PipelineRun not found"
        assert saved_run.status == "completed"
        assert saved_run.violation_count == 2

        saved_case = session.query(Case).filter_by(case_number="ADA-2026-0001").first()
        assert saved_case is not None, "Case not found"
        assert saved_case.violation_class == "new_construction"
        assert saved_case.severity == "critical"
        assert saved_case.pipeline_run.id == "test_run_001"

        print(f"  ✅ PipelineRun: id={saved_run.id}, status={saved_run.status}")
        print(f"  ✅ Case: number={saved_case.case_number}, "
              f"class={saved_case.violation_class}, zone={saved_case.zone_type}")

    engine.dispose()
    print("  ✅ Database models: all assertions passed")


def test_case_evidence_relationship():
    """Insert evidence linked to a case and verify cascade."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from src.database.models import Base, PipelineRun, Case, CaseEvidence

    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        run = PipelineRun(id="test_ev_001", t1_filename="a.tif", t2_filename="b.tif")
        case = Case(
            case_number="ADA-2026-0002", run_id=run.id,
            violation_class="encroachment", confidence=0.85, area_m2=45.0,
        )
        session.add_all([run, case])
        session.flush()

        evidence = CaseEvidence(
            case_id=case.id,
            evidence_type="field_photo",
            file_path="evidence/photo_001.jpg",
            description="Field officer photo of the encroachment",
            uploaded_by="Officer Sharma",
        )
        session.add(evidence)
        session.commit()

        saved_case = session.query(Case).filter_by(case_number="ADA-2026-0002").first()
        assert len(saved_case.evidence) == 1
        assert saved_case.evidence[0].evidence_type == "field_photo"
        assert saved_case.evidence[0].uploaded_by == "Officer Sharma"

        print(f"  ✅ Evidence: type={saved_case.evidence[0].evidence_type}, "
              f"by={saved_case.evidence[0].uploaded_by}")

    engine.dispose()
    print("  ✅ Case-Evidence relationship: all assertions passed")


def test_zone_rules_engine():
    """Verify zone rules severity computation and compliance checks."""
    from src.services.zone_rules import ZoneRulesEngine

    rules = ZoneRulesEngine()

    # Test 1: Heritage + new construction + large area → critical
    sev = rules.compute_severity(
        zone_type="heritage",
        violation_class="new_construction",
        area_m2=150.0,
    )
    assert sev == "critical", f"Expected critical, got {sev}"
    print(f"  ✅ Heritage + new construction (150m²) → {sev}")

    # Test 2: Residential + minor change + small area → low
    sev = rules.compute_severity(
        zone_type="residential",
        violation_class="other_change",
        area_m2=10.0,
    )
    assert sev == "low", f"Expected low, got {sev}"
    print(f"  ✅ Residential + other change (10m²) → {sev}")

    # Test 3: Setback compliance check
    check = rules.check_setback_compliance("heritage", 2.0)
    assert check["compliant"] is False
    assert check["required_m"] == 6.0
    print(f"  ✅ Heritage setback: required={check['required_m']}m, "
          f"measured=2.0m → compliant={check['compliant']}")

    # Test 4: Zone restriction — heritage + new construction
    restriction = rules.check_zone_restriction("heritage", "new_construction")
    assert restriction["restricted"] is True
    assert restriction["severity_override"] == "critical"
    print(f"  ✅ Heritage restriction: {restriction['reason']}")

    # Test 5: List zone types
    zones = rules.list_zone_types()
    assert len(zones) >= 5
    print(f"  ✅ Zone types: {len(zones)} configured")

    print("  ✅ Zone rules engine: all assertions passed")


def main():
    print("🔍 Phase 1 Smoke Test\n")
    print("  Testing database models...")
    test_database_models()

    print("\n  Testing case-evidence relationship...")
    test_case_evidence_relationship()

    print("\n  Testing zone rules engine...")
    test_zone_rules_engine()

    print("\n" + "=" * 50)
    print("🎉 Phase 1: all tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
