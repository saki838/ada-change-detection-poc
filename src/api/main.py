"""
FastAPI wrapper around the 6-stage ADA change-detection pipeline.

Run it:
    uvicorn src.api.main:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/docs        <- interactive Swagger UI
    http://127.0.0.1:8000/redoc       <- alternative ReDoc UI

Endpoints:
    POST /api/v1/detect          Upload T1/T2 (+ optional parcel/red-zone) -> run the pipeline
    GET  /api/v1/runs/{run_id}   List output files for a previous run
    GET  /api/v1/files/{run_id}/{filename}   Download a specific output file
    GET  /api/v1/health          Liveness check

This mirrors exactly what `python -m src.pipeline` does on the command line —
the API is a thin HTTP layer on top of the same `run_pipeline()` function, so
behavior (including the --dummy stand-in models) is identical either way.
"""
from __future__ import annotations

import pathlib
import shutil
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.auth import router as auth_router
from src.api.cases import router as cases_router
from src.api.i18n import router as i18n_router
from src.api.notifications import router as notifications_router
from src.pipeline import load_config, force_dummy_mode, run_pipeline

app = FastAPI(
    title="ADA Change Detection API",
    description=(
        "Upload a baseline (T1) and current drone (T2) image of a site, plus "
        "a parcel boundary, and get back a measured, classified violation "
        "report — change mask, per-region polygons, violation class, area "
        "in m\u00b2, encroachment/setback flags, GeoJSON + overlay image.\n\n"
        "This wraps the same 6-stage pipeline used from the command line "
        "(`python -m src.pipeline`) — see the project README for the full "
        "pipeline architecture."
    ),
    version="0.1.0",
)

OUTPUTS_ROOT = pathlib.Path("outputs/api_runs")
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ViolationSummary(BaseModel):
    class_name: str
    confidence: float
    area_m2: float
    encroaches_parcel: bool
    encroachment_area_m2: float
    setback_violation: bool
    min_setback_m: Optional[float]
    red_zone_overlap: bool
    red_zone_overlap_m2: float


class DetectResponse(BaseModel):
    run_id: str
    violation_count: int
    violations: list[ViolationSummary]
    geojson_url: str
    overlay_png_url: str
    summary_csv_url: str


@app.get("/api/v1/health", tags=["meta"], summary="Liveness check")
def health():
    """Returns OK if the API process is up. Does not verify model weights are present."""
    return {"status": "ok"}


@app.post(
    "/api/v1/detect",
    tags=["detection"],
    summary="Run change detection on an uploaded T1/T2 pair",
    response_model=DetectResponse,
)
async def detect(
    t1: UploadFile = File(..., description="Baseline / approved-plan raster (GeoTIFF)"),
    t2: UploadFile = File(..., description="Current drone raster (GeoTIFF)"),
    parcel: UploadFile = File(..., description="Parcel boundary (GeoJSON)"),
    red_zone: Optional[UploadFile] = File(
        None, description="Optional red-zone boundary (GeoJSON)"
    ),
    dummy: bool = Query(
        False,
        description=(
            "If true, use the no-weights stand-in models (fast image-diff + "
            "box-as-polygon) instead of real BIT/SAM2 — useful for testing "
            "the API itself without waiting on real inference."
        ),
    ),
):
    """
    Runs the full pipeline: align \u2192 change detection \u2192 region extraction
    \u2192 segmentation + classification \u2192 measurement \u2192 report, exactly as
    `python -m src.pipeline` does, and returns the violation results plus
    links to download the generated GeoJSON, overlay image, and summary CSV.

    Note: with real models (dummy=false) this can take anywhere from tens of
    seconds to several minutes on CPU, depending on image size and whether
    SAM2 is enabled — the request will block until it's done.
    """
    run_id = uuid.uuid4().hex[:12]
    run_dir = OUTPUTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        t1_path = await _save_upload(t1, tmp_path / "t1.tif")
        t2_path = await _save_upload(t2, tmp_path / "t2.tif")
        parcel_path = await _save_upload(parcel, tmp_path / "parcel.geojson")
        red_zone_path = (
            await _save_upload(red_zone, tmp_path / "red_zone.geojson")
            if red_zone is not None
            else None
        )

        cfg = load_config(str(DEFAULT_CONFIG_PATH))
        if dummy:
            cfg = force_dummy_mode(cfg)

        try:
            run_pipeline(
                str(t1_path), str(t2_path), str(parcel_path), str(run_dir), cfg,
                red_zone_path=str(red_zone_path) if red_zone_path else None,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e

    violations = _read_summary(run_dir / "summary.csv")

    return DetectResponse(
        run_id=run_id,
        violation_count=len(violations),
        violations=violations,
        geojson_url=f"/api/v1/files/{run_id}/violations.geojson",
        overlay_png_url=f"/api/v1/files/{run_id}/overlay.png",
        summary_csv_url=f"/api/v1/files/{run_id}/summary.csv",
    )


@app.get("/api/v1/runs/{run_id}", tags=["detection"], summary="List output files for a run")
def list_run(run_id: str):
    run_dir = OUTPUTS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return {"run_id": run_id, "files": sorted(p.name for p in run_dir.iterdir())}


@app.get(
    "/api/v1/files/{run_id}/{filename}",
    tags=["detection"],
    summary="Download a generated output file",
)
def get_file(run_id: str, filename: str):
    file_path = OUTPUTS_ROOT / run_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(file_path)


async def _save_upload(upload: UploadFile, dest: pathlib.Path) -> pathlib.Path:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _read_summary(csv_path: pathlib.Path) -> list[ViolationSummary]:
    import pandas as pd

    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    df = df.where(pd.notnull(df), None)
    return [ViolationSummary(**row) for row in df.to_dict(orient="records")]


# Mount the Web-GIS dashboard as static files
DASHBOARD_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

# Redirect root to the dashboard
from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False, tags=["meta"], summary="Redirect to dashboard")
def root():
    return RedirectResponse(url="/dashboard/index.html")


# Register sub-routers
app.include_router(auth_router)
app.include_router(cases_router)
app.include_router(i18n_router)
app.include_router(notifications_router)
