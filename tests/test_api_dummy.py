"""
API smoke test: uploads a synthetic T1/T2/parcel set through the FastAPI
/detect endpoint (dummy mode, no GPU/weights required) and checks the
response + downloadable files, the same way tests/test_pipeline_dummy.py
checks the pipeline directly.

Run with: pytest tests/test_api_dummy.py -v
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.api.main import app
from scripts.run_demo import make_synthetic_pair


def test_detect_endpoint_dummy_mode(tmp_path, monkeypatch):
    # Keep this run's outputs isolated under tmp_path rather than the repo's
    # real outputs/api_runs/, so repeated test runs don't pile up files.
    import src.api.main as api_main

    monkeypatch.setattr(api_main, "OUTPUTS_ROOT", tmp_path / "api_runs")
    api_main.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    t1_path, t2_path, parcel_path = make_synthetic_pair(tmp_path, size=256)

    client = TestClient(app)

    with open(t1_path, "rb") as f1, open(t2_path, "rb") as f2, open(parcel_path, "rb") as fp:
        files = {
            "t1": ("t1.tif", f1, "image/tiff"),
            "t2": ("t2.tif", f2, "image/tiff"),
            "parcel": ("parcel.geojson", fp, "application/geo+json"),
        }
        resp = client.post("/api/v1/detect?dummy=true", files=files)

    assert resp.status_code == 200
    body = resp.json()
    assert body["violation_count"] >= 1
    assert body["violations"][0]["encroaches_parcel"] is True

    run_id = body["run_id"]

    overlay_resp = client.get(f"/api/v1/files/{run_id}/overlay.png")
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"

    list_resp = client.get(f"/api/v1/runs/{run_id}")
    assert list_resp.status_code == 200
    assert "violations.geojson" in list_resp.json()["files"]

    missing_resp = client.get("/api/v1/files/does-not-exist/overlay.png")
    assert missing_resp.status_code == 404


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
