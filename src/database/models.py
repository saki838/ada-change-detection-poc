"""SQLAlchemy ORM models for the ADA change-detection system.

Tables map to the UPLC system spec:
  - pipeline_runs      → track each AI pipeline execution
  - cases              → core violation case management
  - case_evidence      → field photos, drone images, documents
  - enforcement_notices → generated PDF notices
  - sanctioned_plans   → approved building plans for permit reconciliation
  - zone_boundaries    → zone type boundaries + rule config

Designed SQLite-first for the demo, with a migration path to PostGIS:
  - Geometry stored as GeoJSON text (works in both SQLite and PostGIS)
  - All IDs are integers with autoincrement where sensible
  - Timestamps use timezone-naive UTC (SQLite-compatible)
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

from src.database.connection import get_engine

Base = declarative_base()


# ── Enums (stored as strings for SQLite compatibility) ──────────────

class ViolationClass:
    """Taxonomy from UPLC scope doc Section 4/6."""
    NEW_CONSTRUCTION = "new_construction"
    HORIZONTAL_EXPANSION = "horizontal_expansion"
    VERTICAL_EXPANSION = "vertical_expansion"
    ENCROACHMENT = "encroachment"
    UNAUTHORIZED_PAVING = "unauthorized_paving"
    VEGETATION_CLEARANCE = "vegetation_clearance"
    OTHER = "other_change"

    CHOICES = [
        NEW_CONSTRUCTION, HORIZONTAL_EXPANSION, VERTICAL_EXPANSION,
        ENCROACHMENT, UNAUTHORIZED_PAVING, VEGETATION_CLEARANCE, OTHER,
    ]


class ZoneType:
    """Zone classifications from the Agra Development Authority scope."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    HERITAGE = "heritage"
    GREEN_BELT = "green_belt"
    RIVERFRONT = "riverfront"
    INDUSTRIAL = "industrial"
    OTHER = "other"

    CHOICES = [
        RESIDENTIAL, COMMERCIAL, HERITAGE,
        GREEN_BELT, RIVERFRONT, INDUSTRIAL, OTHER,
    ]


class Severity:
    """Violation severity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    CHOICES = [LOW, MEDIUM, HIGH, CRITICAL]


class CaseStatus:
    """Lifecycle of a violation case from detection through resolution."""
    DETECTED = "detected"
    ASSIGNED = "assigned"
    FIELD_VERIFIED = "field_verified"
    ENFORCEMENT_READY = "enforcement_ready"
    NOTICE_ISSUED = "notice_issued"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

    CHOICES = [
        DETECTED, ASSIGNED, FIELD_VERIFIED, ENFORCEMENT_READY,
        NOTICE_ISSUED, RESOLVED, ESCALATED,
    ]


class EvidenceType:
    """Types of evidence that can be attached to a case."""
    FIELD_PHOTO = "field_photo"
    DRONE_IMAGE = "drone_image"
    PIPELINE_OVERLAY = "pipeline_overlay"
    DOCUMENT = "document"
    OTHER = "other"

    CHOICES = [
        FIELD_PHOTO, DRONE_IMAGE, PIPELINE_OVERLAY, DOCUMENT, OTHER,
    ]


class NoticeStatus:
    """Lifecycle of an enforcement notice."""
    DRAFT = "draft"
    ISSUED = "issued"
    SERVED = "served"

    CHOICES = [DRAFT, ISSUED, SERVED]


# ── Tables ──────────────────────────────────────────────────────────

class PipelineRun(Base):
    """Tracks each execution of the AI change-detection pipeline."""
    __tablename__ = "pipeline_runs"

    id = Column(String(32), primary_key=True)  # short UUID hex
    t1_filename = Column(String(255), nullable=False)
    t2_filename = Column(String(255), nullable=False)
    parcel_filename = Column(String(255), nullable=True)
    red_zone_filename = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default="running")
    violation_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime, nullable=True)

    cases = relationship("Case", back_populates="pipeline_run", cascade="all, delete-orphan")


class Case(Base):
    """A detected violation that has been promoted to a tracked case."""
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_number = Column(String(32), unique=True, nullable=False, index=True)
    run_id = Column(String(32), ForeignKey("pipeline_runs.id"), nullable=False)

    # Detection results
    violation_class = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    area_m2 = Column(Float, nullable=False)
    location_geojson = Column(Text, nullable=True)  # Point or Polygon

    # Zone & severity
    zone_type = Column(String(20), nullable=True)
    severity = Column(String(10), nullable=False, default="medium")

    # Workflow
    status = Column(String(20), nullable=False, default="detected")
    assigned_to = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    pipeline_run = relationship("PipelineRun", back_populates="cases")
    evidence = relationship("CaseEvidence", back_populates="case", cascade="all, delete-orphan")
    notices = relationship("EnforcementNotice", back_populates="case", cascade="all, delete-orphan")


class CaseEvidence(Base):
    """Evidence attachments linked to a case."""
    __tablename__ = "case_evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    evidence_type = Column(String(20), nullable=False)
    file_path = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    uploaded_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    uploaded_by = Column(String(255), nullable=True)

    case = relationship("Case", back_populates="evidence")


class EnforcementNotice(Base):
    """Generated enforcement notice PDFs."""
    __tablename__ = "enforcement_notices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    notice_number = Column(String(32), unique=True, nullable=False, index=True)
    status = Column(String(10), nullable=False, default="draft")
    file_path = Column(String(512), nullable=True)
    legal_reference = Column(Text, nullable=True)
    generated_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    issued_at = Column(DateTime, nullable=True)
    served_at = Column(DateTime, nullable=True)
    served_to = Column(String(255), nullable=True)

    case = relationship("Case", back_populates="notices")


class SanctionedPlan(Base):
    """Approved building plans for permit reconciliation (Stage 4.5)."""
    __tablename__ = "sanctioned_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parcel_id = Column(String(64), nullable=False, index=True)
    plan_number = Column(String(64), unique=True, nullable=False)
    approved_footprint_geojson = Column(Text, nullable=False)  # Polygon
    approved_area_m2 = Column(Float, nullable=False)
    approved_height_m = Column(Float, nullable=True)
    approved_floors = Column(Integer, nullable=True)
    approval_date = Column(DateTime, nullable=False)
    expiry_date = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False, default="active")  # active, expired, revoked
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class UserRole:
    """User roles for the enforcement system."""
    ADMIN = "admin"
    SUPERVISOR = "supervisor"
    ENFORCEMENT_OFFICER = "enforcement_officer"

    CHOICES = [ADMIN, SUPERVISOR, ENFORCEMENT_OFFICER]

    PERMISSIONS = {
        ADMIN: ["cases:read", "cases:write", "cases:assign", "cases:delete",
                "users:read", "users:write", "users:delete", "reports:read",
                "system:config"],
        SUPERVISOR: ["cases:read", "cases:write", "cases:assign",
                     "users:read", "reports:read"],
        ENFORCEMENT_OFFICER: ["cases:read", "cases:write", "reports:read"],
    }


class User(Base):
    """System user account with role-based access control."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default=UserRole.ENFORCEMENT_OFFICER)
    phone = Column(String(20), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class AuditLog(Base):
    """Audit trail for all user actions on cases and system resources."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(64), nullable=False)  # e.g. case.created, case.status_updated
    resource_type = Column(String(32), nullable=False)  # case, user, notice
    resource_id = Column(String(32), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ZoneBoundary(Base):
    """Zone boundary definitions for rule-based enforcement."""
    __tablename__ = "zone_boundaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_type = Column(String(20), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    boundary_geojson = Column(Text, nullable=False)  # Polygon / MultiPolygon
    setback_distance_m = Column(Float, nullable=False, default=3.0)
    severity_boost = Column(Float, nullable=False, default=1.0)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ── Helpers ─────────────────────────────────────────────────────────

def init_db(db_path: str = "data/ada.db") -> str:
    """Create all tables and return the database path.

    Safe to call multiple times — uses IF NOT EXISTS semantics.
    """
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return db_path
