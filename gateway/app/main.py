"""FastAPI app: CORS, router mounts, /health, and demo-admin seed on startup."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import ensure_seed_admin
from app.config import get_settings
from app.db import SessionLocal
from app.routers import auth as auth_router
from app.routers import detect as detect_router

logger = logging.getLogger("gateway")
settings = get_settings()

app = FastAPI(title="ADA Encroachment Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(auth_router.router, prefix="/api/auth", tags=["auth"])
app.include_router(detect_router.router, prefix="/api", tags=["detect"])


@app.on_event("startup")
def _seed() -> None:
    """Best-effort demo-admin seed (admin/admin123); never blocks startup."""
    try:
        db = SessionLocal()
        try:
            ensure_seed_admin(db)
        finally:
            db.close()
    except Exception as exc:  # pragma: no cover - startup resilience
        logger.warning("seed admin skipped: %s", exc)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
