# ADA Encroachment — AI Change Detection POC

Self-hosted, offline-at-inference pipeline that takes a baseline image (T1) and a
current drone image (T2) of the same site and produces a measured, classified
violation report — no China-origin models, zero licensing cost.

This repo is the code scaffold for the pipeline described in the POC doc. It is
built to run on a real GPU workstation (RTX 3090/3060, 16–32GB RAM, Ubuntu 22.04,
Python 3.10/3.11, PyTorch 2.2+, CUDA 12.x) — **not** in this sandbox, which has no
GPU and no access to model-weight hosts (Hugging Face, GitHub releases, etc).

Everything is wired end-to-end and runnable *today* in "dummy" mode (a lightweight
image-diff stand-in for BIT, and a simple contour-based stand-in for SAM), so you
can prove the plumbing works on day one — see [Definition of Done](#definition-of-done).
Swap in real weights when ready and nothing else changes.

## Pipeline (matches Section 3 of the POC doc)

```
Stage 0  Input prep         GDAL/rasterio        T1+T2 rasters       -> aligned pair
Stage 1  Change detection   BIT / ChangeFormer    aligned pair        -> binary change mask
Stage 2  Region extraction  OpenCV                change mask         -> bounding boxes
Stage 3  Seg + classify     SAM 2.1 + RT-DETR      boxes               -> polygon + class
Stage 4  Measurement        shapely + GSD          polygons + parcel   -> area, encroachment, setback
Stage 5  Report              Python                measured violations -> GeoJSON + overlay + table
```

Each stage is a standalone module in `src/` with a plain-Python function
signature, so Track A (data + models) and Track B (glue + geo + report) can be
built independently and meet at the handoffs, per Section 5 of the doc.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Smoke test the whole pipeline on synthetic data (no weights, no GPU needed)
python scripts/run_demo.py --dummy

# 2. Once you have real data + weights, run for real:
python -m src.pipeline \
    --t1 data/site/baseline.tif \
    --t2 data/site/drone_current.tif \
    --parcel data/site/parcel_boundary.geojson \
    --out outputs/run_001 \
    --weights weights/bit_levir.pt
```

## API (Swagger UI)

The same pipeline is also exposed over HTTP via FastAPI, with an
auto-generated interactive Swagger UI.

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/docs** in a browser \u2014 you'll get an
interactive page listing every endpoint, with a "Try it out" button to
upload files and run a detection directly from the browser, no curl/Postman
needed. ReDoc (a read-only alternative view) is at `/redoc`.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/detect` | Upload T1/T2 (+ optional parcel/red-zone) \u2014 runs the full pipeline, returns violation results + file URLs |
| GET | `/api/v1/runs/{run_id}` | List output files for a previous run |
| GET | `/api/v1/files/{run_id}/{filename}` | Download a specific output file (GeoJSON / overlay PNG / summary CSV) |
| GET | `/api/v1/health` | Liveness check |

`/api/v1/detect` takes a `dummy` query parameter (`?dummy=true`) to use the
no-weights stand-in models instead of real BIT/SAM2 \u2014 useful for testing
the API itself without waiting on real inference. This mirrors the
`--dummy` flag on the command-line pipeline exactly.

## Getting real weights and data

- `scripts/download_levir_cd.py` — fetches the LEVIR-CD+ public bridge dataset
  (aerial building change pairs, via Hugging Face `datasets`) used to stand the
  pipeline up before site data is ready, per Section 2 of the doc. Pure
  Python — works on Windows/macOS/Linux, no bash required.
- `scripts/download_weights.py` — prints setup steps for pretrained BIT /
  SAM 2.1 / RT-DETR checkpoints. **Run this on your workstation**, not in a
  locked-down sandbox — it needs Hugging Face / GitHub release access.
- Loading discipline (per the doc): all checkpoints are loaded with
  `torch.load(..., weights_only=True)` against `.safetensors` where the upstream
  repo provides them. See `src/utils/safe_load.py`.

## Repo layout

```
src/
  stage0_align.py          Stage 0 — GDAL/rasterio alignment
  stage1_change_detection.py  Stage 1 — BIT / ChangeFormer / dummy diff
  stage2_region_extraction.py Stage 2 — OpenCV contours -> boxes
  stage3_segment_classify.py  Stage 3 — SAM 2.1 + RT-DETR / dummy
  stage4_measurement.py    Stage 4 — shapely area / encroachment / setback
  stage5_report.py         Stage 5 — GeoJSON + overlay PNG + summary table
  pipeline.py              Orchestrator — wires all 6 stages together
  models/                  Model wrapper classes (real + dummy)
  utils/                   Geo helpers, viz, safe checkpoint loading
scripts/
  run_demo.py              End-to-end smoke test on synthetic data
  download_levir_cd.py     Pulls public LEVIR-CD+ bridge dataset
  download_weights.py      Prints BIT / SAM2.1 / RT-DETR checkpoint setup steps
tests/
  test_pipeline_dummy.py   Pytest smoke test (no GPU/weights required)
```

## Definition of Done (POC acceptance, per Section 6)

- [ ] Feed one (T1, T2) pair for a real site -> get back, with no manual steps:
      change mask, per-region polygons, violation class, area in m², GeoJSON +
      overlay image.
- [ ] Whole run completes offline on the entry workstation.
- [ ] Change-detection F1 >= 0.85 on held-out validation pairs (LEVIR-CD or site
      data).
- [ ] A 10-minute demo run, using Section 3's flow as the talking track.

## Status of this scaffold

| Stage | Plumbing (dummy mode) | Real model wired |
|---|---|---|
| 0 Align | done | done (rasterio, no "model" to swap) |
| 1 Change detection | done (diff-based stand-in) | wrapper ready for BIT / ChangeFormer weights |
| 2 Region extraction | done | done (OpenCV, no weights needed) |
| 3 Segment + classify | done (contour polygon + rule-based label) | wrapper ready for SAM 2.1 / RT-DETR weights |
| 4 Measurement | done | done (shapely, no weights needed) |
| 5 Report | done | done |

Next step once you confirm the labeled (T1, T2, mask) data question from Section 2:
run `scripts/download_weights.py` and `scripts/download_levir_cd.py` on the
workstation, then flip `--dummy` off.
