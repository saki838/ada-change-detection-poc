"""Core BFF routes under /api: detect, list runs, run detections, run mask."""
from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, Response
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import get_settings
from app.db import get_db
from app.inference_client import call_predict
from app.models import Detection, Run, User
from app.schemas import DetectResponse, RunListResponse, RunSummary

router = APIRouter()

_ALLOWED_CT = {"image/tiff", "image/png", "image/jpeg"}
_EXT_BY_CT = {"image/tiff": "tif", "image/png": "png", "image/jpeg": "jpg"}


def _ext_for(upload: UploadFile) -> str:
    if upload.filename and "." in upload.filename:
        return upload.filename.rsplit(".", 1)[-1].lower()
    return _EXT_BY_CT.get(upload.content_type or "", "bin")


def _bad_pair() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="t1/t2 required or size mismatch",
    )


@router.post("/detect", status_code=201, response_model=DetectResponse)
async def detect(
    t1: UploadFile | None = File(None),
    t2: UploadFile | None = File(None),
    pixel_size_m: float = Form(10.0),
    mode: str = Form("ml"),
    name: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DetectResponse:
    settings = get_settings()

    # 1. Validate uploads --------------------------------------------------
    if t1 is None or t2 is None:
        raise _bad_pair()
    if t1.content_type not in _ALLOWED_CT or t2.content_type not in _ALLOWED_CT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupported content-type (expect image/tiff|png|jpeg)",
        )
    t1_bytes = await t1.read()
    t2_bytes = await t2.read()
    if not t1_bytes or not t2_bytes:
        raise _bad_pair()
    if mode not in {"ml", "diff"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="mode must be ml or diff"
        )

    # 2. Persist raw images to the shared volume ---------------------------
    run_uid = uuid.uuid4().hex
    run_dir = Path(settings.image_store_dir) / "runs" / run_uid
    run_dir.mkdir(parents=True, exist_ok=True)
    t1_path = run_dir / f"t1.{_ext_for(t1)}"
    t2_path = run_dir / f"t2.{_ext_for(t2)}"
    t1_path.write_bytes(t1_bytes)
    t2_path.write_bytes(t2_bytes)

    # 3. Create run row (status=running) -----------------------------------
    run = Run(
        user_id=user.id,
        name=name,
        mode=mode,
        status="running",
        t1_path=str(t1_path),
        t2_path=str(t2_path),
        pixel_size_m=pixel_size_m,
        crs="EPSG:4326",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # 4. Call inference ----------------------------------------------------
    try:
        result = await call_predict(
            t1_bytes,
            t2_bytes,
            pixel_size_m=pixel_size_m,
            mode=mode,
            threshold=0.5,
            min_area_px=20,
        )
    except httpx.HTTPStatusError as exc:
        run.status = "failed"
        run.error = f"inference HTTP {exc.response.status_code}"
        db.commit()
        if exc.response.status_code == 400:
            raise _bad_pair()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="inference unavailable"
        )
    except httpx.HTTPError as exc:
        run.status = "failed"
        run.error = f"inference transport error: {exc}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="inference unavailable"
        )

    # 5. Persist mask ------------------------------------------------------
    mask_b64 = result.get("mask_png_b64")
    if mask_b64:
        mask_path = run_dir / "mask.png"
        mask_path.write_bytes(base64.b64decode(mask_b64))
        run.mask_path = str(mask_path)

    # 6. Persist detections ------------------------------------------------
    polygons = result.get("polygons") or []
    for poly in polygons:
        geojson = json.dumps(poly["geometry"])
        db.add(
            Detection(
                run_id=run.id,
                geom=func.ST_SetSRID(func.ST_GeomFromGeoJSON(geojson), 4326),
                area_m2=float(poly.get("area_m2", 0.0)),
                confidence=float(poly.get("confidence", 0.0)),
                class_label="change",
            )
        )

    # 7. Finalize run ------------------------------------------------------
    run.status = "completed"
    run.num_detections = int(result.get("num_detections", len(polygons)))
    run.total_area_m2 = float(result.get("total_area_m2", 0.0))
    inf_ms = result.get("inference_ms")
    run.inference_ms = float(inf_ms) if inf_ms is not None else None
    run.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)

    # 8. Respond -----------------------------------------------------------
    return DetectResponse(
        run_id=run.id,
        status=run.status,
        created_at=run.created_at,
        mode=run.mode,
        num_detections=run.num_detections,
        total_area_m2=run.total_area_m2,
        mask_png_url=f"/api/runs/{run.id}/mask.png",
        detections_url=f"/api/runs/{run.id}/detections",
    )


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RunListResponse:
    total = db.scalar(
        select(func.count()).select_from(Run).where(Run.user_id == user.id)
    )
    rows = db.scalars(
        select(Run)
        .where(Run.user_id == user.id)
        .order_by(Run.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    runs = [
        RunSummary(
            run_id=r.id,
            name=r.name,
            status=r.status,
            mode=r.mode,
            num_detections=r.num_detections,
            total_area_m2=r.total_area_m2,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return RunListResponse(runs=runs, total=int(total or 0))


def _owned_run(db: Session, run_id: int, user: User) -> Run:
    run = db.scalar(select(Run).where(Run.id == run_id, Run.user_id == user.id))
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
        )
    return run


@router.get("/runs/{run_id}/detections")
def run_detections(
    run_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    _owned_run(db, run_id, user)
    rows = db.execute(
        text(
            "SELECT id, ST_AsGeoJSON(geom) AS geojson, area_m2, confidence "
            "FROM detections WHERE run_id = :rid ORDER BY id"
        ),
        {"rid": run_id},
    ).all()
    features = [
        {
            "type": "Feature",
            "geometry": json.loads(row.geojson),
            "properties": {
                "detection_id": row.id,
                "area_m2": row.area_m2,
                "confidence": row.confidence,
            },
        }
        for row in rows
    ]
    return JSONResponse({"type": "FeatureCollection", "features": features})


@router.get("/runs/{run_id}/mask.png")
def run_mask(
    run_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    run = _owned_run(db, run_id, user)
    if not run.mask_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="mask not found"
        )
    path = Path(run.mask_path)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="mask not found"
        )
    return Response(content=path.read_bytes(), media_type="image/png")
