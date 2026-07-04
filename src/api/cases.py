"""FastAPI router for case management endpoints.

All endpoints are under /api/v1/cases and require the database to be
initialized. The database path is configured via the DB_PATH constant.

Endpoints:
    GET    /api/v1/cases          — List/filter cases
    GET    /api/v1/cases/stats    — Aggregate case statistics
    GET    /api/v1/cases/{id}     — Get case detail
    PATCH  /api/v1/cases/{id}/status   — Update case status
    POST   /api/v1/cases/{id}/assign   — Assign case to officer
    POST   /api/v1/cases/{id}/evidence — Add evidence to case
    GET    /api/v1/cases/{id}/notice   — Generate & download notice PDF
"""
from __future__ import annotations

import io
import os
import pathlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.auth import _require_auth, _require_role
from src.database.connection import get_engine
from src.database.models import Case, PipelineRun, UserRole, init_db

from src.services.case_service import CaseService
from src.services.notice_generator import generate_notice, save_notice_record

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])

# Default database path — can be overridden by setting the ADA_DB_PATH env var
DB_PATH = os.getenv("ADA_DB_PATH", "data/ada.db")

# Initialize database at module load (safe: CREATE TABLE IF NOT EXISTS)
init_db(DB_PATH)

# ── Pydantic schemas ────────────────────────────────────────────────


class CaseResponse(BaseModel):
    id: int
    case_number: str
    run_id: str
    violation_class: str
    confidence: float
    area_m2: float
    zone_type: Optional[str]
    severity: str
    status: str
    assigned_to: Optional[str]
    description: Optional[str]
    notes: Optional[str]
    created_at: str  # ISO format
    updated_at: str  # ISO format


class CaseListResponse(BaseModel):
    total: int
    cases: list[CaseResponse]


class CaseStatsResponse(BaseModel):
    total: int
    by_status: dict
    by_severity: dict
    by_zone: dict


class CreateCaseRequest(BaseModel):
    """Schema for creating a case via the API (for demo seeding / testing)."""
    violation_class: str
    confidence: float = 0.9
    area_m2: float
    zone_type: str = "other"
    severity: str = "medium"
    status: str = "detected"
    assigned_to: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None
    lat: float = 27.1767
    lng: float = 78.0081


class StatusUpdateRequest(BaseModel):
    status: str
    notes: Optional[str] = None


class AssignRequest(BaseModel):
    officer_name: str


class EvidenceResponse(BaseModel):
    id: int
    case_id: int
    evidence_type: str
    file_path: str
    description: Optional[str]
    uploaded_by: Optional[str]
    uploaded_at: str
    image_url: Optional[str] = None


class CaseDetailResponse(CaseResponse):
    """Extended case detail with evidence and notices."""
    evidence: list[EvidenceResponse] = []


class BulkNoticeRequest(BaseModel):
    """Schema for bulk notice generation — optional filters."""
    status: Optional[str] = None
    severity: Optional[str] = None
    zone_type: Optional[str] = None
    limit: int = 200


# ── Helper: get db session ──────────────────────────────────────────


def _get_session():
    """Get a SQLAlchemy session with the database initialized."""
    engine = get_engine(DB_PATH)
    return Session(engine)


def _case_to_response(case) -> CaseResponse:
    """Convert a Case ORM object to a CaseResponse schema."""
    return CaseResponse(
        id=case.id,
        case_number=case.case_number,
        run_id=case.run_id,
        violation_class=case.violation_class,
        confidence=case.confidence,
        area_m2=case.area_m2,
        zone_type=case.zone_type,
        severity=case.severity,
        status=case.status,
        assigned_to=case.assigned_to,
        description=case.description,
        notes=case.notes,
        created_at=case.created_at.isoformat() if case.created_at else "",
        updated_at=case.updated_at.isoformat() if case.updated_at else "",
    )


def _evidence_to_response(evidence) -> EvidenceResponse:
    """Convert a CaseEvidence ORM object to a response schema."""
    return EvidenceResponse(
        id=evidence.id,
        case_id=evidence.case_id,
        evidence_type=evidence.evidence_type,
        file_path=evidence.file_path,
        description=evidence.description,
        uploaded_by=evidence.uploaded_by,
        uploaded_at=evidence.uploaded_at.isoformat() if evidence.uploaded_at else "",
        image_url=f"/api/v1/cases/{evidence.case_id}/evidence/{evidence.id}/image",
    )


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("", response_model=CaseListResponse, summary="List cases with optional filters")
def list_cases(
    user=Depends(_require_auth),
    status: Optional[str] = Query(None, description="Filter by status"),
    zone_type: Optional[str] = Query(None, description="Filter by zone type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    assigned_to: Optional[str] = Query(None, description="Filter by assigned officer"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Results offset"),
):
    """List cases with optional filters. Most recent cases first."""
    with _get_session() as session:
        service = CaseService()
        cases = service.list_cases(
            session,
            status=status,
            zone_type=zone_type,
            severity=severity,
            assigned_to=assigned_to,
            limit=limit,
            offset=offset,
        )
        total = service.count_cases(
            session,
            status=status,
            zone_type=zone_type,
            severity=severity,
        )
    return CaseListResponse(
        total=total,
        cases=[_case_to_response(c) for c in cases],
    )


@router.post("/create", response_model=CaseResponse, summary="Create a new case (for demo seeding / testing)")
def create_case(
    body: CreateCaseRequest,
    admin=Depends(_require_role(UserRole.ADMIN, UserRole.SUPERVISOR)),
):
    """Create a single case via the API. Used for demo seeding and testing.

    Creates a PipelineRun placeholder internally if no runs exist.
    """
    import json
    import uuid

    with _get_session() as session:
        # Find or create a demo pipeline run
        run = session.query(PipelineRun).order_by(PipelineRun.created_at.desc()).first()
        if run is None:
            run = PipelineRun(
                id=uuid.uuid4().hex[:8],
                t1_filename="api/demo_t1.tif",
                t2_filename="api/demo_t2.tif",
                parcel_filename="api/demo_parcel.geojson",
                status="completed",
                violation_count=0,
                created_at=datetime.now(timezone.utc),
            )
            session.add(run)
            session.flush()

        service = CaseService()

        # Count existing cases to generate a case number
        total = service.count_cases(session)
        case_number = f"ADA-API-{total + 1:04d}"

        case = Case(
            case_number=case_number,
            run_id=run.id,
            violation_class=body.violation_class,
            confidence=body.confidence,
            area_m2=body.area_m2,
            location_geojson=json.dumps({
                "type": "Point",
                "coordinates": [body.lng, body.lat],
            }),
            zone_type=body.zone_type,
            severity=body.severity,
            status=body.status,
            assigned_to=body.assigned_to,
            description=body.description,
            notes=body.notes,
        )
        session.add(case)
        session.commit()
        session.refresh(case)

    return _case_to_response(case)


@router.get("/stats", response_model=CaseStatsResponse, summary="Get aggregate case statistics")
def get_stats(
    user=Depends(_require_auth),
):
    """Return aggregate statistics about cases (counts by status, severity, zone)."""
    with _get_session() as session:
        service = CaseService()
        stats = service.get_case_stats(session)
    return CaseStatsResponse(**stats)


@router.get("/{case_id}", response_model=CaseDetailResponse, summary="Get case detail with evidence")
def get_case(
    case_id: int,
    user=Depends(_require_auth),
):
    """Get full details for a single case, including evidence and notices."""
    with _get_session() as session:
        service = CaseService()
        case = service.get_case(session, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
        base = _case_to_response(case)
        evidence = [_evidence_to_response(e) for e in case.evidence]
        return CaseDetailResponse(**base.model_dump(), evidence=evidence)


@router.patch(
    "/{case_id}/status",
    response_model=CaseResponse,
    summary="Update case workflow status",
)
def update_status(
    case_id: int,
    body: StatusUpdateRequest,
    user=Depends(_require_auth),
):
    """Update the workflow status of a case (e.g. detected → assigned → field_verified)."""
    with _get_session() as session:
        service = CaseService()
        case = service.update_status(
            session, case_id, body.status, notes=body.notes,
        )
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
        return _case_to_response(case)


@router.post(
    "/{case_id}/assign",
    response_model=CaseResponse,
    summary="Assign case to an enforcement officer",
)
def assign_case(
    case_id: int,
    body: AssignRequest,
    supervisor=Depends(_require_role(UserRole.ADMIN, UserRole.SUPERVISOR)),
):
    """Assign a case to a named enforcement officer for field verification."""
    with _get_session() as session:
        service = CaseService()
        case = service.assign_case(session, case_id, body.officer_name)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
        return _case_to_response(case)


@router.post(
    "/{case_id}/evidence",
    response_model=EvidenceResponse,
    summary="Upload evidence (photo/document) to a case",
)
async def add_evidence(
    case_id: int,
    file: UploadFile = File(..., description="Evidence file (photo, document, etc.)"),
    evidence_type: str = Query("field_photo", description="Type of evidence"),
    description: Optional[str] = Query(None, description="Optional description"),
    uploaded_by: Optional[str] = Query(None, description="Uploader name"),
    user=Depends(_require_auth),
):
    """Upload an evidence file (photo, document, etc.) and attach it to a case.

    The file is saved to `outputs/evidence/{case_id}/` and a record
    is created in the case_evidence table.
    """
    case_dir = pathlib.Path(f"outputs/evidence/{case_id}")
    case_dir.mkdir(parents=True, exist_ok=True)

    # Save file with timestamp prefix to avoid name collisions
    # Sanitize filename to prevent path traversal
    safe_filename = pathlib.Path(file.filename).name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = case_dir / f"{timestamp}_{safe_filename}"

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    with _get_session() as session:
        service = CaseService()
        evidence = service.add_evidence(
            session,
            case_id=case_id,
            evidence_type=evidence_type,
            file_path=str(dest),
            description=description,
            uploaded_by=uploaded_by,
        )
        if evidence is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
        return _evidence_to_response(evidence)


@router.get(
    "/{case_id}/evidence/{evidence_id}/image",
    summary="Get evidence image (returns placeholder if file not found)",
)
def get_evidence_image(case_id: int, evidence_id: int):
    """Return an evidence image file, or a generated placeholder if the file
    doesn't exist yet (useful for demo cases without real imagery)."""
    from fastapi.responses import Response

    with _get_session() as session:
        from src.database.models import CaseEvidence

        ev = session.query(CaseEvidence).filter_by(
            id=evidence_id, case_id=case_id
        ).first()
        if ev is None:
            raise HTTPException(status_code=404, detail="Evidence not found")

        # Try to serve actual file
        if ev.file_path and pathlib.Path(ev.file_path).exists():
            return FileResponse(
                ev.file_path,
                media_type="image/jpeg",
            )

    # Generate a placeholder SVG on-the-fly
    label = (ev.evidence_type or "evidence").replace("_", " ").title()
    desc = (ev.description or "Site imagery")[:60]
    date_str = ev.uploaded_at.strftime("%d %b %Y") if ev.uploaded_at else ""

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
  <rect width="400" height="300" fill="#1F3355" rx="8"/>
  <rect x="10" y="10" width="380" height="280" fill="#2E4A78" rx="6"/>
  <text x="200" y="80" font-family="Arial,sans-serif" font-size="48" fill="#D4834F" text-anchor="middle">🛰️</text>
  <text x="200" y="130" font-family="Arial,sans-serif" font-size="16" fill="#F7F3EC" text-anchor="middle" font-weight="bold">{label}</text>
  <text x="200" y="160" font-family="Arial,sans-serif" font-size="13" fill="#AAB8D4" text-anchor="middle">{desc}</text>
  <text x="200" y="200" font-family="Arial,sans-serif" font-size="11" fill="#8899B4" text-anchor="middle">Case #{case_id} | Evidence #{evidence_id}</text>
  <text x="200" y="220" font-family="Arial,sans-serif" font-size="11" fill="#8899B4" text-anchor="middle">{date_str} | Uploaded by: {ev.uploaded_by or 'system'}</text>
  <text x="200" y="260" font-family="Arial,sans-serif" font-size="11" fill="#6B7B99" text-anchor="middle">Upload actual images via POST /cases/{{id}}/evidence</text>
</svg>"""

    return Response(content=svg, media_type="image/svg+xml")


@router.get(
    "/{case_id}/notice",
    summary="Generate and download enforcement notice PDF",
)
def get_notice(
    case_id: int,
    user=Depends(_require_auth),
):
    """Generate an enforcement notice PDF for a case and download it.

    The PDF is generated once and cached in `outputs/notices/`. Subsequent
    requests return the cached file unless regeneration is forced.
    """
    with _get_session() as session:
        service = CaseService()
        case = service.get_case(session, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

        notices_dir = pathlib.Path("outputs/notices")
        notices_dir.mkdir(parents=True, exist_ok=True)

        # Check if notice already generated
        if case.notices:
            latest_notice = case.notices[-1]
            if latest_notice.file_path and pathlib.Path(latest_notice.file_path).exists():
                return FileResponse(
                    latest_notice.file_path,
                    media_type="application/pdf",
                    filename=f"notice_{case.case_number}.pdf",
                )

        # Generate new notice PDF
        pdf_path = generate_notice(case, str(notices_dir))

        # Save record to database
        save_notice_record(session, case, pdf_path)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"notice_{case.case_number}.pdf",
        )


@router.post(
    "/notices/bulk",
    summary="Generate enforcement notice PDFs for multiple cases (returns ZIP)",
)
def bulk_generate_notices(
    body: BulkNoticeRequest,
    admin=Depends(_require_role(UserRole.ADMIN, UserRole.SUPERVISOR)),
):
    """Generate enforcement notice PDFs for all cases matching the given
    filters and return them as a bundled ZIP file.

    Filters (all optional):
      - status:   Filter by case status (e.g. "detected", "assigned")
      - severity: Filter by severity ("low", "medium", "high", "critical")
      - zone_type: Filter by zone ("heritage", "riverfront", "green_belt", etc.)
      - limit:    Maximum number of notices to generate (default 200)
    """
    import json
    import zipfile
    from fastapi.responses import StreamingResponse

    notices_dir = pathlib.Path("outputs/notices")
    notices_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    manifest = []

    with _get_session() as session:
        service = CaseService()
        cases = service.list_cases(
            session,
            status=body.status,
            severity=body.severity,
            zone_type=body.zone_type,
            limit=body.limit,
        )

        if not cases:
            raise HTTPException(status_code=404, detail="No cases found matching the given filters")

        # Build ZIP in memory while session is still open
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for case in cases:
                try:
                    # Check if a notice already exists
                    existing_pdf = None
                    if case.notices:
                        latest = case.notices[-1]
                        if latest.file_path and pathlib.Path(latest.file_path).exists():
                            existing_pdf = latest.file_path

                    if existing_pdf:
                        pdf_path = existing_pdf
                    else:
                        pdf_path = generate_notice(case, str(notices_dir))
                        save_notice_record(session, case, pdf_path)

                    # Add PDF to the ZIP
                    zip_name = f"notice_{case.case_number.lower().replace('-', '_')}.pdf"
                    zf.write(pdf_path, arcname=zip_name)
                    manifest.append({
                        "case_number": case.case_number,
                        "case_id": case.id,
                        "status": "generated",
                    })
                except Exception:
                    manifest.append({
                        "case_number": case.case_number,
                        "case_id": case.id,
                        "status": "failed",
                    })

            # Add summary manifest
            zf.writestr("manifest.json", json.dumps({
                "total": len(cases),
                "generated": len([m for m in manifest if m["status"] == "generated"]),
                "failed": len([m for m in manifest if m["status"] == "failed"]),
                "cases": manifest,
            }, indent=2))

    buf.seek(0)

    # Build a descriptive filename
    filter_parts = []
    if body.status:
        filter_parts.append(body.status)
    if body.severity:
        filter_parts.append(body.severity)
    if body.zone_type:
        filter_parts.append(body.zone_type.replace("_", "-"))
    filter_tag = "_".join(filter_parts) + "_" if filter_parts else ""
    zip_filename = f"ada_notices_{filter_tag}{len(cases)}cases.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
        },
    )
