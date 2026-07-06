# ADA Encroachment Detection — MVP

Containerized change-detection demo: authenticated T1/T2 orthophoto upload → ML (or non-ML `diff`) inference → change polygons with `area_m2` + `confidence` → PostGIS → Leaflet map. Four microservices: `frontend` (5173), `gateway` (8000, the only public backend), `inference` (8001, internal), `db` (5432, PostGIS 16-3.4). Non-China weight posture: the only pretrained file is torchvision ResNet34 ImageNet from download.pytorch.org.

**Full per-file build blueprint:** see [`BUILD_GUIDE.md`](./BUILD_GUIDE.md) (sections 00 architecture → 90 infra).

## Quickstart

```bash
cp .env.example .env          # then set a strong JWT_SECRET
make download-weights         # cache the non-China ResNet34 ImageNet encoder
docker compose up --build     # or: make up   (frontend :5173, gateway :8000)
```

Then open http://localhost:5173, log in as the seeded demo user, and upload a T1/T2 pair. The gateway health check is at http://localhost:8000/health. Run `make test` for the end-to-end smoke test, `make down` to stop, `make clean` to also wipe volumes.
