"""SQLAlchemy 2 engine, session factory, declarative Base, and DB dependency.

GeoAlchemy2 is imported here for its side effects so that PostGIS ``geometry``
columns are recognised on this engine.
"""
from __future__ import annotations

from typing import Iterator

import geoalchemy2  # noqa: F401  (side-effect: registers the Geometry type)
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for the ORM models (maps existing PostGIS tables)."""


engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, class_=Session
)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yield a session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
