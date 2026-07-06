"""ORM classes mapping the three PostGIS tables (users, runs, detections).

Columns mirror db/init/01_schema.sql (the locked db_schema) 1:1. The gateway
does not create these tables; the postgres image runs the DDL at init.
"""
from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="analyst")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    runs: Mapped[list["Run"]] = relationship(
        "Run", back_populates="user", cascade="all, delete-orphan"
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="ml")
    t1_path: Mapped[str] = mapped_column(Text, nullable=False)
    t2_path: Mapped[str] = mapped_column(Text, nullable=False)
    mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    pixel_size_m: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)
    crs: Mapped[str | None] = mapped_column(String(32), nullable=True)
    num_detections: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_area_m2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped["User"] = relationship("User", back_populates="runs")
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="run", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    geom: Mapped[object] = mapped_column(
        Geometry(geometry_type="POLYGON", srid=4326), nullable=False
    )
    area_m2: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    class_label: Mapped[str] = mapped_column(
        String(64), nullable=False, default="change"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped["Run"] = relationship("Run", back_populates="detections")
