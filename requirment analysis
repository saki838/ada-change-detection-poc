# ADA Encroachment — AI Change Detection System

**Agra Development Authority — Automated Building Violation Detection & Enforcement**

A self-hosted, offline-at-inference AI pipeline that takes a baseline image (T1) and a current drone/satellite image (T2) of the same site and produces a measured, classified violation report with enforcement-grade documentation. Designed for the Agra Development Authority (ADA), India, covering heritage, residential, green belt, riverfront, commercial, and industrial zones.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Pipeline](#architecture--pipeline)
- [System Components](#system-components)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Datasets & Models](#datasets--models)
- [Project Structure](#project-structure)
- [Definition of Done](#definition-of-done)
- [Production Roadmap](#production-roadmap)
- [Comprehensive Requirements](#comprehensive-requirements)
- [License](#license)

---

## Overview

| Attribute | Detail |
|---|---|
| **Domain** | Smart city enforcement, urban planning, geospatial AI |
| **Pipeline Stages** | 6 stages: Align → Detect → Extract → Segment/Classify → Measure → Report |
| **Core AI Models** | BIT (change detection), SAM 2.1 (segmentation), RT-DETR (classification) |
| **Languages** | Python 3.10/3.11 (backend), JavaScript (dashboard) |
| **API Framework** | FastAPI with auto-generated Swagger UI & ReDoc |
| **Database** | SQLite (dev/POC) → PostGIS (production) |
| **Target Hardware** | NVIDIA RTX 3090/4090, 32GB+ RAM, Ubuntu 22.04 |
| **Violation Classes** | New construction, horizontal/vertical expansion, encroachment, unauthorized paving, vegetation clearance |

The system compares a **baseline (T1)** image — either a prior survey or the approved building plan — against a **current (T2)** drone orthomosaic. It identifies physical changes, classifies them by violation type, measures area and setback compliance, reconciles against sanctioned building permits, and generates enforcement-ready outputs (GeoJSON, overlay imagery, summary reports, and legal notice PDFs).

**Key design principles:**
- **No China-origin models** — uses only Western/open-source model architectures
- **Zero licensing cost** — all models are MIT/Apache/BSD licensed
- **Offline inference** — runs entirely on-premises; no internet required at inference time
- **Safe checkpoint loading** — `torch.load(..., weights_only=True)` with `.safetensors` preference
- **Human-in-the-loop** — all AI detections require field verification before enforcement action

---

## Architecture & Pipeline

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│   T1 (Baseline/Plan)    T2 (Drone/Satellite)    Parcel Boundary     │
│       (GeoTIFF)              (GeoTIFF)           (GeoJSON/SHP)      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 0 — Input Alignment          │  rasterio / GDAL              │
│  Reproject & resample T1/T2 to      │  target resolution: 0.5m/px   │
│  shared CRS, extent, pixel grid     │  bilinear resampling          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Change Detection         │  BIT / ChangeFormer           │
│  Compare aligned pair → binary      │  tile + stitch for large      │
│  change mask (changed pixels)       │  images (256px tiles)         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2 — Region Extraction        │  OpenCV                       │
│  Contour finding → bounding boxes   │  min area filter (150px)      │
│  per discrete change region         │  morphology cleaning          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3 — Segmentation + Classify  │  SAM 2.1 + RT-DETR           │
│  Box-prompted SAM → precise polygon │  7 violation classes          │
│  RT-DETR → violation type label     │  confidence scoring           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4 — Measurement & Rules      │  shapely / GeoPandas          │
│  Area (m²), parcel encroachment,    │  configurable setback rules   │
│  setback distance, red-zone overlap │  by zone type                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4.5 — Permit Reconciliation  │  SQLAlchemy / Shapely         │
│  Compare detected violations        │  excess area detection        │
│  against sanctioned building plans  │  unapproved footprint flags   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 5 — Report Generation        │  GeoPandas / Pillow / fpdf2  │
│  GeoJSON violations layer           │  overlay PNG (change mask     │
│  Summary CSV table                  │  on T2)                       │
│  Enforcement Notice PDF             │                               │
└─────────────────────────────────────────────────────────────────────┘
```

### System Architecture (Production)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Drone /   │────▶│   Nginx /    │────▶│  FastAPI App     │
│   Satellite │     │   ALB        │     │  (Docker)        │
│   Upload    │     └──────────────┘     └────────┬─────────┘
└─────────────┘                                   │
                                                  ▼
                                   ┌──────────────────────────┐
                                   │   Celery Workers (GPU)   │
                                   │  - BIT inference         │
                                   │  - SAM2 segmentation     │
                                   │  - RT-DETR classification│
                                   └────────┬─────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────┐
                    ▼                       ▼                   ▼
           ┌──────────────┐       ┌──────────────┐    ┌──────────────┐
           │   PostGIS    │       │   Redis      │    │  MinIO/S3    │
           │  (Database)  │       │ (Cache/Queue)│    │ (Artifacts)  │
           └──────────────┘       └──────────────┘    └──────────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │  Dashboard SPA   │
                                   │  (Leaflet +      │
                                   │   Chart.js)      │
                                   └──────────────────┘
```

---

## System Components

### 6-Stage Pipeline

Each stage is a standalone module in `src/` with a plain-Python function signature:

| Stage | Module | Tool/Model | Input | Output |
|---|---|---|---|---|
| **0 — Alignment** | `src/stage0_align.py` | rasterio / GDAL | T1 + T2 rasters | Aligned RGB pair (shared CRS/extent) |
| **1 — Change Detection** | `src/stage1_change_detection.py` | BIT / ChangeFormer | Aligned pair | Binary change mask |
| **2 — Region Extraction** | `src/stage2_region_extraction.py` | OpenCV | Change mask | Bounding boxes per region |
| **3 — Seg + Classify** | `src/stage3_segment_classify.py` | SAM 2.1 + RT-DETR | Boxes + T2 | Polygon + violation class |
| **4 — Measurement** | `src/stage4_measurement.py` | shapely + GSD | Polygons + parcel | Area, encroachment, setback |
| **4.5 — Permit Rec.** | `src/stage4_5_permit_reconciliation.py` | SQLAlchemy + shapely | Violations + DB | Permit compliance status |
| **5 — Report** | `src/stage5_report.py` | GeoPandas + Pillow + fpdf2 | Violations | GeoJSON + overlay + CSV + PDF |

### API Server (FastAPI)

The pipeline is exposed over HTTP via FastAPI at `src/api/main.py` with:
- JWT-based authentication with role-based access control
- Case management (CRUD, status workflow, assignment)
- Evidence upload and management
- Enforcement notice PDF generation (bulk + single)
- Multi-language support (English + Hindi)
- Notification system (email/SMS with demo mode)
- Auto-generated interactive Swagger UI at `/docs`
- PWA-ready dashboard with offline service worker

### Dashboard (SPA)

A single-page application at `dashboard/index.html` with:
- Interactive Leaflet map with marker clustering, heatmap, and zone overlays
- Real-time case list with filter/search/progress bars
- Analytics charts (severity, zone, status distributions, weekly trends)
- Case detail modal with evidence gallery, timeline, and notice download
- Role-based UI (officers see only their assigned cases)
- Multi-language (English / Hindi) with client-side translation
- PWA support (manifest + service worker for offline caching)
- Responsive design (desktop + mobile)

### Database (SQLAlchemy ORM)

Models in `src/database/models.py` include:
- **PipelineRun** — tracks each AI execution
- **Case** — violation case with lifecycle (detected → assigned → field_verified → enforcement_ready → notice_issued → resolved/escalated)
- **CaseEvidence** — field photos, drone imagery, overlays
- **EnforcementNotice** — generated notice PDFs with legal references
- **SanctionedPlan** — approved building plans for permit reconciliation
- **User** — role-based accounts (admin, supervisor, enforcement_officer)
- **AuditLog** — full audit trail for compliance
- **ZoneBoundary** — zone type polygon definitions with setback rules

---

## Quick Start

### Prerequisites

- **Hardware:** GPU workstation (RTX 3090/3060 recommended, CPU-only works for dummy mode)
- **Software:** Python 3.10/3.11, pip, venv
- **OS:** Ubuntu 22.04 (Windows via WSL2 also works)

### Installation

```bash
# Clone the repository
git clone <repo-url> ada-change-detection
cd ada-change-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAM 2.1 (for real segmentation — skip for dummy mode)
git clone https://github.com/facebookresearch/sam2.git
pip install -e sam2
```

### Smoke Test (No GPU, No Weights)

```bash
# Run the full pipeline on synthetic data with dummy stand-in models
python scripts/run_demo.py --dummy

# Check outputs in outputs/demo_run/
#   violations.geojson  — detected violations with geometry
#   overlay.png         — change mask overlaid on T2
#   summary.csv         — tabular violation summary
```

### Run with Real Models

```bash
# 1. Download weights
python scripts/download_weights.py  # prints instructions

# 2. Download LEVIR-CD+ bridge dataset
python scripts/download_levir_cd.py

# 3. Run pipeline
python -m src.pipeline \
    --t1 data/site/baseline.tif \
    --t2 data/site/drone_current.tif \
    --parcel data/site/parcel_boundary.geojson \
    --out outputs/run_001 \
    --weights weights/bit_levir.pt
```

### Run as API Server

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --port 8000

# Open in browser:
#   http://127.0.0.1:8000/docs     — Interactive Swagger UI
#   http://127.0.0.1:8000/redoc    — ReDoc documentation
#   http://127.0.0.1:8000/         — Web dashboard
```

### Seed Demo Data

```bash
# Seed users for the dashboard
python scripts/seed_users.py

# Seed demo violation cases, evidence, zone boundaries, sanctioned plans
python scripts/seed_demo_data.py

# (Optional) Seed demo data via API instead of direct DB access
python scripts/seed_demo_via_api.py

# (Optional) Generate evidence images (synthetic or LEVIR-CD based)
python scripts/seed_evidence_via_api.py          # synthetic
python scripts/seed_evidence_via_api.py --use-levir  # real aerial imagery

# Login with:
#   Admin:      admin@ada.gov.in     / admin123
#   Supervisor: supervisor@ada.gov.in / super123
#   Officer:    sharma@ada.gov.in    / field123
```

---

## API Reference

### Core Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| **POST** | `/api/v1/detect` | — | Upload T1/T2 + parcel → run full pipeline, return violations |
| **GET** | `/api/v1/runs/{run_id}` | — | List output files for a previous run |
| **GET** | `/api/v1/files/{run_id}/{filename}` | — | Download output file (GeoJSON/PNG/CSV) |
| **GET** | `/api/v1/health` | — | Liveness check |

### Auth Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| **POST** | `/api/v1/auth/login` | — | Authenticate → JWT token |
| **GET** | `/api/v1/auth/me` | Any | Current user profile |
| **POST** | `/api/v1/auth/users` | Admin | Create new user |
| **GET** | `/api/v1/auth/users` | Admin/Supervisor | List all users |
| **PATCH** | `/api/v1/auth/users/{id}/role` | Admin | Update user role |
| **POST** | `/api/v1/auth/users/{id}/deactivate` | Admin | Deactivate user |

### Case Management Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| **GET** | `/api/v1/cases` | Any | List/filter cases |
| **GET** | `/api/v1/cases/stats` | Any | Aggregate case statistics |
| **GET** | `/api/v1/cases/{id}` | Any | Case detail with evidence |
| **PATCH** | `/api/v1/cases/{id}/status` | Officer+ | Update case status |
| **POST** | `/api/v1/cases/{id}/assign` | Admin/Supervisor | Assign to officer |
| **POST** | `/api/v1/cases/{id}/evidence` | Officer+ | Upload evidence file |
| **GET** | `/api/v1/cases/{id}/notice` | Officer+ | Download notice PDF |
| **POST** | `/api/v1/cases/notices/bulk` | Admin/Supervisor | Bulk notice ZIP |

### i18n Endpoints

| Method | Path | Purpose |
|---|---|---|
| **GET** | `/api/v1/i18n/languages` | List supported languages |
| **GET** | `/api/v1/i18n/translations?lang=hi` | Get translations |

### Notification Endpoints

| Method | Path | Purpose |
|---|---|---|
| **GET** | `/api/v1/notifications/log` | Recent notification log |
| **POST** | `/api/v1/notifications/test` | Send test notification |

---

## Dashboard

The web dashboard is served at `http://localhost:8000/dashboard/` and provides:

- **Login Page** — Dark-themed authentication with role-aware redirect
- **Stats Bar** — At-a-glance totals: open cases, critical, high, resolved
- **Interactive Map** — Leaflet-based with:
  - Marker clustering for dense violation areas
  - Color-coded severity markers (green → amber → orange → red)
  - Heatmap overlay for violation density
  - Popup with case details on click
  - Layer toggles: All / Critical / Heat
- **Case Panel** — Searchable, filterable case list with:
  - Severity badges, progress bars, status dots
  - Quick-assign buttons for unassigned cases
  - Filter tabs with counts (All, Detected, Assigned, Critical, Heritage, Resolved)
- **Case Detail Modal** — Tabbed interface:
  - **Info tab**: Violation class, zone, area, confidence, status, description
  - **Evidence tab**: Image gallery with lightbox viewer
  - **Timeline tab**: Case lifecycle visualization
  - **Notice tab**: Generate/download enforcement notice PDF
- **Analytics Panel** — Chart.js graphs:
  - Cases by severity (doughnut)
  - Cases by zone (bar)
  - Cases by status (horizontal bar)
  - Weekly trend (line)
- **Multi-Language** — Toggle between English and Hindi with full translation
- **PWA** — Service worker for offline caching, installable manifest
- **Responsive** — Works on desktop, tablet, and mobile

---

## Datasets & Models

### Current Datasets

| Dataset | Source | Format | Size | Purpose |
|---|---|---|---|---|
| **LEVIR-CD+** | Hugging Face `blanchon/LEVIR_CDPlus` | PNG (A/B/mask) 1024×1024 | ~985 pairs | Bridge training for change detection |
| **Synthetic Samples** | `scripts/create_sample_images.py` | GeoTIFF + PNG + GeoJSON | Configurable | Local testing/validation |
| **Site Samples** | `data/site/samples/` | GeoTIFF + GeoJSON | 6 scenarios × 3 samples | Demo & smoke tests |

### Supported Model Architectures

| Model | Stage | Source | Weights | Size |
|---|---|---|---|---|
| **BIT** (Bitemporal Image Transformer) | 1 — Change detection | [justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD) | Google Drive link in repo | ~50 MB |
| **ChangeFormer** | 1 — Change detection (higher accuracy) | [wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) | Not yet wired | ~80 MB |
| **SAM 2.1** (Hiera Tiny) | 3 — Segmentation | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | [dl.fbaipublicfiles.com](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | ~150 MB |
| **RT-DETR** (R50vd) | 3 — Classification | Hugging Face `PekingU/rtdetr_r50vd_coco_o365` | HF transformers | ~200 MB |

### Dummy Mode Stand-ins

For environments without GPU or internet access, the pipeline includes lightweight stand-ins:

| Stage | Real Model | Dummy Stand-in |
|---|---|---|
| 1 — Change detection | BIT / ChangeFormer | `DummyDiffChangeDetector` — Gaussian blur + pixel difference + threshold |
| 3 — Segmentation | SAM 2.1 | `DummyContourSegmenter` — returns box as rectangle polygon |
| 3 — Classification | RT-DETR | `DummyRuleClassifier` — labels everything as "change" |

Swap to real models by flipping `config.yaml` — zero code changes needed.

### Violation Classes

| Class | Description |
|---|---|
| `new_construction` | Unauthorized structure where none existed |
| `horizontal_expansion` | Building extended sideways beyond approved footprint |
| `vertical_expansion` | Additional floors beyond approved height |
| `encroachment` | Construction crossing parcel boundary |
| `unauthorized_paving` | Unapproved impervious surface |
| `vegetation_clearance` | Unauthorized tree/vegetation removal |
| `other_change` | Unclassified change |

---

## Project Structure

```
ada-change-detection/
├── config.yaml                    # Central pipeline configuration
├── requirements.txt               # Python dependencies
├── .gitignore / .gitattributes    # VCS configuration
│
├── src/                           # Python source code
│   ├── __init__.py
│   ├── pipeline.py                # Orchestrator: wires all 6 stages
│   │
│   ├── stage0_align.py            # GDAL/rasterio alignment
│   ├── stage1_change_detection.py # BIT / ChangeFormer / dummy diff
│   ├── stage2_region_extraction.py# OpenCV contours → bounding boxes
│   ├── stage3_segment_classify.py # SAM 2.1 + RT-DETR / dummy
│   ├── stage4_measurement.py      # shapely area / encroachment / setback
│   ├── stage4_5_permit_reconciliation.py  # Compare vs sanctioned plans
│   ├── stage5_report.py           # GeoJSON + overlay PNG + summary CSV
│   │
│   ├── models/                    # Model wrapper classes
│   │   ├── change_detector.py     # BIT + DummyDiff wrappers
│   │   └── segment_classify.py    # SAM2 + RT-DETR + Dummy wrappers
│   │
│   ├── api/                       # FastAPI server
│   │   ├── main.py                # App setup, detect endpoint, file serving
│   │   ├── auth.py                # JWT auth, user management
│   │   ├── cases.py               # Case CRUD, evidence, notices, bulk
│   │   ├── i18n.py                # Multi-language support
│   │   └── notifications.py       # Notification log & test
│   │
│   ├── database/                  # SQLAlchemy ORM
│   │   ├── connection.py          # Engine factory (SQLite → PostGIS)
│   │   └── models.py              # All ORM models + enums
│   │
│   ├── services/                  # Business logic
│   │   ├── auth_service.py        # Password hashing, JWT, user CRUD
│   │   ├── case_service.py        # Case lifecycle, severity, statistics
│   │   ├── notice_generator.py    # PDF enforcement notice generation
│   │   ├── notification_service.py# Email/SMS notifications
│   │   ├── zone_rules.py          # Zone-based severity & compliance rules
│   │   └── translation_service.py # i18n translation dictionary
│   │
│   └── utils/                     # Shared utilities
│       ├── geo.py                 # Pixel ↔ world transforms, GSD lookup
│       ├── safe_load.py           # Safe checkpoint loading (weights_only)
│       └── viz.py                 # Overlay & annotation helpers
│
├── dashboard/                     # Web dashboard (Single Page App)
│   ├── index.html                 # Full SPA with CSS + JavaScript
│   ├── manifest.json              # PWA manifest
│   ├── sw.js                      # Service worker (offline caching)
│   └── icon-192.svg / icon-512.svg # App icons
│
├── data/                          # Datasets & site data
│   ├── levir_cd/                  # LEVIR-CD+ dataset (after download)
│   │   ├── train/ (A/, B/, label/)
│   │   ├── val/   (A/, B/, label/)
│   │   └── test/  (A/, B/, label/)
│   └── site/                      # Site-specific data
│       ├── samples/               # Synthetic sample images
│       │   ├── new_construction/
│       │   ├── demolition/
│       │   ├── vegetation_clearance/
│       │   ├── horizontal_expansion/
│       │   ├── unauthorized_paving/
│       │   └── encroachment/      # Includes parcel.geojson
│       └── _synthetic_demo/       # Generated by run_demo.py
│
├── scripts/                       # Utility & demo scripts
│   ├── run_demo.py                # End-to-end smoke test
│   ├── create_sample_images.py    # Generate synthetic T1/T2 pairs
│   ├── download_levir_cd.py       # Download LEVIR-CD+ dataset
│   ├── download_weights.py        # Print weight download instructions
│   ├── seed_users.py              # Seed admin/supervisor/officer users
│   ├── seed_demo_data.py          # Seed demo cases + evidence + zones
│   ├── seed_demo_via_api.py       # Seed via HTTP API (no DB access)
│   ├── seed_evidence_via_api.py   # Generate & upload evidence images
│   ├── test_on_levir_sample.py    # Test pipeline on real LEVIR imagery
│   ├── test_phase1.py             # DB models + zone rules validation
│   ├── test_phase2.py             # Permit reconciliation + case service
│   ├── test_phase3.py             # API endpoint integration tests
│   └── run_sample_generator.ps1   # PowerShell runner for sample gen
│
├── tests/                         # Pytest test suite
│   ├── test_pipeline_dummy.py     # Pipeline smoke test (no GPU)
│   └── test_api_dummy.py          # API endpoint smoke test
│
├── outputs/                       # Generated outputs (gitignored)
│   ├── api_runs/                  # API-triggered pipeline outputs
│   ├── evidence/                  # Uploaded evidence files
│   ├── notices/                   # Generated enforcement notice PDFs
│   └── demo_run/                  # Demo pipeline outputs
│
└── weights/                       # Model weights (gitignored)
    ├── bit_levir.pt               # BIT checkpoint
    ├── sam2.1_tiny.pt             # SAM 2.1 checkpoint
    └── rtdetr_violations.pt       # Fine-tuned RT-DETR
```

---

## Definition of Done

### POC Acceptance Criteria

- [ ] Feed one (T1, T2) pair for a real site → get back, with no manual steps: change mask, per-region polygons, violation class, area in m², GeoJSON + overlay image
- [ ] Whole run completes offline on the entry workstation (RTX 3090, 32GB RAM)
- [ ] Change-detection F1 ≥ 0.85 on held-out validation pairs (LEVIR-CD or site data)
- [ ] A 10-minute demo run using the full pipeline flow

### Model Status

| Stage | Plumbing (dummy mode) | Real Model Wired |
|---|---|---|
| 0 — Align | ✅ Done | ✅ Done (rasterio, no "model" to swap) |
| 1 — Change Detection | ✅ Done (diff-based) | ⏳ Wrapper ready for BIT / ChangeFormer weights |
| 2 — Region Extraction | ✅ Done | ✅ Done (OpenCV, no weights needed) |
| 3 — Segment + Classify | ✅ Done (contour + rule) | ⏳ Wrapper ready for SAM 2.1 / RT-DETR weights |
| 4 — Measurement | ✅ Done | ✅ Done (shapely, no weights needed) |
| 4.5 — Permit Reconciliation | ✅ Done | ✅ Done (DB-driven, no weights) |
| 5 — Report | ✅ Done | ✅ Done |

---

## Production Roadmap

### Phase 0 — POC Scaffold (✅ Complete)
- [x] Full 6-stage pipeline with orchestrator
- [x] Dummy mode for smoke testing (no GPU/weights)
- [x] FastAPI server with Swagger UI
- [x] Synthetic data generator
- [x] LEVIR-CD+ bridge dataset downloader
- [x] Unit tests for pipeline + API

### Phase 1 — Real Models & Validation (2–4 weeks)
- [ ] Download BIT weights → achieve F1 ≥ 0.85 on LEVIR-CD
- [ ] Download SAM 2.1 weights → verify segmentation quality
- [ ] Fine-tune RT-DETR on violation classes
- [ ] Run pipeline with real models end-to-end
- [ ] Validate against held-out LEVIR-CD test set

### Phase 2 — Case Management & Reporting (4–6 weeks)
- [x] Permit reconciliation (Stage 4.5)
- [x] Case service with lifecycle management
- [x] Enforcement notice PDF generation
- [ ] Automated severity scoring with zone rules
- [x] Bulk notice generation (ZIP)

### Phase 3 — Auth & API (4–6 weeks)
- [x] JWT authentication with role-based access
- [x] User management (admin, supervisor, officer)
- [x] Case CRUD endpoints
- [x] Evidence upload + management
- [x] Dashboard with interactive map + analytics

### Phase 4 — Production Infrastructure (4–8 weeks)
- [ ] Migrate SQLite → PostGIS with Alembic migrations
- [ ] Containerize with Docker + Docker Compose
- [ ] Set up async pipeline with Celery + Redis
- [ ] Implement object storage (MinIO/S3) for artifacts
- [ ] Add rate limiting, load balancing (Nginx)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Implement MLOps tracking (MLflow)

### Phase 5 — Site-Specific Training (8–12 weeks)
- [ ] Collect and label Agra-specific T1/T2 violation pairs
- [ ] Fine-tune BIT on Agra data
- [ ] Fine-tune RT-DETR classification head
- [ ] Field validation campaign
- [ ] Establish continuous training pipeline

### Phase 6 — Production Deployment (4–8 weeks)
- [ ] Deploy to GPU cluster (on-prem or cloud)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure backup & disaster recovery
- [ ] Implement audit logging for compliance
- [ ] Security audit & penetration testing
- [ ] Load testing (Locust/k6)

### Phase 7 — Continuous Improvement (Ongoing)
- [ ] Add new violation classes
- [ ] Improve model accuracy with active learning
- [ ] Expand to new zone types
- [ ] Real-time drone feed integration
- [ ] Mobile field app for officers
- [ ] WhatsApp/SMS notification integration

---

## Comprehensive Requirements

### Hardware Requirements

| Environment | GPU | RAM | Storage | CPU |
|---|---|---|---|---|
| **Dev/POC** | RTX 3090 (24GB) or RTX 3060 (12GB) | 32–64 GB | 1–2 TB NVMe SSD | 8+ cores |
| **Production (on-prem)** | 2–4× RTX 4090 / A5000 (24GB) | 64 GB/node | 4 TB+ NVMe RAID | 16+ cores |
| **Cloud (AWS)** | g4dn.xlarge (T4) → g5.2xlarge (A10G) | 32–64 GB | EBS gp3 2TB | 4–8 vCPU |

### Software Stack

| Category | POC (Current) | Production Target |
|---|---|---|
| **Language** | Python 3.10/3.11 | Python 3.11+ |
| **ML Framework** | PyTorch 2.2+ | PyTorch 2.2+ + TensorRT |
| **Database** | SQLite | PostGIS + PgBouncer |
| **API Server** | FastAPI + uvicorn | FastAPI + gunicorn + Nginx |
| **Async Processing** | Synchronous | Celery + Redis / RabbitMQ |
| **Containerization** | — | Docker + Kubernetes |
| **CI/CD** | — | GitHub Actions |
| **Monitoring** | — | Prometheus + Grafana + Sentry |
| **Object Storage** | Local filesystem | MinIO / AWS S3 |
| **Caching** | — | Redis |
| **MLOps** | — | MLflow / W&B |

### Security & Compliance

- **Indian data sovereignty** — All infrastructure within India
- **Encryption at rest** — AES-256 for imagery, DB, model weights
- **Encryption in transit** — TLS 1.3 for all API traffic
- **RBAC** — Admin, Supervisor, Enforcement Officer roles with permissions
- **Audit logging** — Every case update, login, file access recorded
- **Data retention** — Raw imagery: 5 years; Cases: 10 years
- **Model governance** — Versioned checkpoints, explainability (Grad-CAM)
- **Legal framework** — U.P. Urban Planning Act 1973, AMASR Act 1958, EPA 1986, NGT guidelines

### Team Requirements

| Role | Count | Skills |
|---|---|---|
| ML Engineer | 2–3 | PyTorch, remote sensing, computer vision |
| Backend Developer | 2–3 | FastAPI, PostGIS, Celery, Docker |
| Frontend Developer | 1–2 | React/Next.js, Leaflet, Mapbox |
| GIS Specialist | 1–2 | GDAL, rasterio, spatial analysis |
| DevOps Engineer | 1–2 | K8s, GPU orchestration, CI/CD |
| Domain Expert | 1 | Urban planning, building codes |
| QA Engineer | 1 | Automated testing, model evaluation |
| Product Manager | 1 | Stakeholder management, govt relations |

### Estimated Annual Costs

| Category | Low (On-prem, 2 GPU) | Medium (Hybrid) | High (Cloud, 4 GPU) |
|---|---|---|---|
| Hardware | $15K (one-time) | $30K/yr | $60K/yr |
| Software/Licenses | $5K | $15K | $30K+ |
| Personnel | $120K (4 staff) | $250K (6–8 staff) | $500K+ (10+ staff) |
| Data Acquisition | $10K/yr | $25K/yr | $50K/yr |
| Infrastructure | $5K/yr | $20K/yr | $40K/yr |
| **Total (first year)** | **~$155K** | **~$350K** | **~$705K+** |

---

## Zone Rules

The system enforces zone-specific regulations based on the Agra Master Plan 2031:

| Zone | Base Severity | Severity Boost | Setback | Special Rules |
|---|---|---|---|---|
| **Heritage** | High | 2.0× | 6.0 m | Auto-critical for any construction |
| **Green Belt** | Medium | 1.5× | 5.0 m | New construction prohibited |
| **Riverfront** | High | 1.8× | 10.0 m | Strict buffer enforcement |
| **Residential** | Low | 1.0× | 3.0 m | Standard rules |
| **Commercial** | Medium | 1.2× | 4.0 m | Standard rules |
| **Industrial** | Low | 1.0× | 5.0 m | Standard rules |

---

## License

This project is built entirely with open-source components:

| Component | License |
|---|---|
| BIT (ChangeFormer) | MIT |
| SAM 2.1 | Apache 2.0 |
| RT-DETR | Apache 2.0 |
| PyTorch | BSD |
| FastAPI | MIT |
| All custom code | MIT |

All model weights are loaded with `weights_only=True` or from `.safetensors` format (no arbitrary pickle execution). See `src/utils/safe_load.py` for the loading discipline.

---

*Built for the Agra Development Authority. For questions, contact the project maintainers.*
