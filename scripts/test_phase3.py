"""Phase 3 smoke test — verifies the Case Management API endpoints.

Run this after:
    pip install httpx

Usage:
    python scripts/test_phase3.py

It will:
  1. Start the FastAPI app with TestClient
  2. Test all case management endpoints
  3. Verify proper HTTP status codes and response schemas
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Set test database path
import os
os.environ["ADA_DB_PATH"] = "data/test_phase3.db"


def setup_test_data():
    """Create test cases and a pipeline run in the database."""
    from src.database.connection import get_engine
    from src.database.models import init_db, Base, PipelineRun, Case, SanctionedPlan
    from sqlalchemy.orm import Session

    db_path = os.environ["ADA_DB_PATH"]
    init_db(db_path)
    engine = get_engine(db_path)

    with Session(engine) as session:
        # Pipeline run
        run = PipelineRun(id="phase3_test_run", t1_filename="a.tif", t2_filename="b.tif", status="completed")
        session.add(run)
        session.flush()

        # Add a sanctioned plan
        plan_poly = {"type": "Polygon", "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]}
        plan = SanctionedPlan(
            parcel_id="PARCEL-TEST",
            plan_number="BP-TEST-001",
            approved_footprint_geojson=json.dumps(plan_poly),
            approved_area_m2=500.0,
            approval_date="2026-01-01",
            status="active",
        )
        session.add(plan)

        # Test cases
        cases_data = [
            Case(case_number="ADA-2026-1001", run_id=run.id,
                 violation_class="new_construction", confidence=0.92, area_m2=150.0,
                 zone_type="heritage", severity="critical", status="detected",
                 description="New construction in heritage zone"),
            Case(case_number="ADA-2026-1002", run_id=run.id,
                 violation_class="encroachment", confidence=0.85, area_m2=45.0,
                 zone_type="residential", severity="medium", status="assigned",
                 assigned_to="Officer Sharma",
                 description="Boundary encroachment"),
            Case(case_number="ADA-2026-1003", run_id=run.id,
                 violation_class="vegetation_clearance", confidence=0.78, area_m2=200.0,
                 zone_type="green_belt", severity="high", status="field_verified",
                 description="Vegetation cleared in green belt zone"),
        ]
        for c in cases_data:
            session.add(c)

        session.commit()

    return db_path


def cleanup_test_data():
    """Remove the test database file."""
    db_path = os.environ["ADA_DB_PATH"]
    p = pathlib.Path(db_path)
    if p.exists():
        p.unlink()
    # Also remove WAL and SHM files
    for ext in ["-wal", "-shm"]:
        wal = p.parent / (p.name + ext)
        if wal.exists():
            wal.unlink()


def test_case_api_endpoints():
    """Test all case management API endpoints via TestClient."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)

    # Test GET /api/v1/cases - list all
    response = client.get("/api/v1/cases")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["total"] == 3, f"Expected 3 cases, got {data['total']}"
    assert len(data["cases"]) == 3
    print(f"  ✅ GET /cases: {data['total']} cases listed")

    # Test GET /api/v1/cases with filters
    response = client.get("/api/v1/cases?severity=critical")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1, f"Expected 1 critical case, got {data['total']}"
    print(f"  ✅ GET /cases?severity=critical: {data['total']} case")

    response = client.get("/api/v1/cases?assigned_to=Officer+Sharma")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1, f"Expected 1 assigned case, got {data['total']}"
    print(f"  ✅ GET /cases?assigned_to=Officer+Sharma: {data['total']} case")

    # Test GET /api/v1/cases/stats
    response = client.get("/api/v1/cases/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["by_severity"]["critical"] == 1
    print(f"  ✅ GET /cases/stats: {data['total']} total, {data['by_severity']['critical']} critical")

    # Test GET /api/v1/cases/{id} - get case detail
    response = client.get("/api/v1/cases/1")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["case_number"] == "ADA-2026-1001"
    assert data["severity"] == "critical"
    print(f"  ✅ GET /cases/1: {data['case_number']} ({data['severity']})")

    # Test GET /api/v1/cases/{id} - not found
    response = client.get("/api/v1/cases/999")
    assert response.status_code == 404
    print(f"  ✅ GET /cases/999: 404 not found")

    # Test PATCH /api/v1/cases/{id}/status
    response = client.patch(
        "/api/v1/cases/1/status",
        json={"status": "assigned", "notes": "Assigning for field verification"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "assigned"
    print(f"  ✅ PATCH /cases/1/status: status changed to '{data['status']}'")

    # Test POST /api/v1/cases/{id}/assign
    response = client.post(
        "/api/v1/cases/1/assign",
        json={"officer_name": "Officer Verma"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["assigned_to"] == "Officer Verma"
    print(f"  ✅ POST /cases/1/assign: assigned to {data['assigned_to']}")

    # Test POST /api/v1/cases/{id}/evidence
    response = client.post(
        "/api/v1/cases/1/evidence",
        params={"evidence_type": "field_photo", "description": "Site visit photo", "uploaded_by": "Officer Verma"},
        files={"file": ("photo.jpg", b"fake_image_data", "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_type"] == "field_photo"
    assert data["uploaded_by"] == "Officer Verma"
    print(f"  ✅ POST /cases/1/evidence: evidence #{data['id']} ({data['evidence_type']})")

    print("  ✅ All case API endpoints: assertions passed")


def main():
    print("🔍 Phase 3 Smoke Test\n")

    print("  Setting up test data...")
    db_path = setup_test_data()
    print(f"  ✅ Test database created at {db_path}")

    print("\n  Testing case API endpoints...")
    test_case_api_endpoints()

    print("\n  Cleaning up...")
    cleanup_test_data()
    print("  ✅ Test data removed")

    print("\n" + "=" * 50)
    print("🎉 Phase 3: all tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
