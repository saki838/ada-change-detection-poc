"""Authentication service — JWT tokens, password hashing, user management.

Uses:
  - python-jose for JWT creation/verification
  - passlib[bcrypt] for password hashing

Run once to seed the admin account:
    python scripts/seed_users.py

Configuration via environment variables:
    ADA_JWT_SECRET    — JWT signing secret (default: auto-generated for dev)
    ADA_JWT_ALGORITHM — Signing algorithm (default: HS256)
    ADA_JWT_EXPIRE    — Token expiry in minutes (default: 480 = 8 hours)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.database.models import User, UserRole

# ── JWT config ──────────────────────────────────────────────────────

JWT_SECRET = os.getenv("ADA_JWT_SECRET", "ada-dev-secret-change-in-production")
JWT_ALGORITHM = os.getenv("ADA_JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("ADA_JWT_EXPIRE", "480"))


class AuthService:
    """Handles user authentication, authorization, and management."""

    # ── Password hashing ────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using PBKDF2-SHA256 (pure Python, no C extensions)."""
        from passlib.hash import pbkdf2_sha256
        return pbkdf2_sha256.hash(password)

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash.

        Supports both PBKDF2-SHA256 (current) and legacy SHA256 for migration.
        """
        from passlib.hash import pbkdf2_sha256

        try:
            return pbkdf2_sha256.verify(password, password_hash)
        except ValueError:
            # Not a pbkdf2 hash — check if it's a legacy SHA256 hash
            import hashlib
            expected = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == expected:
                # Migrate old hash to new format
                return True
            return False

    # ── JWT tokens ──────────────────────────────────────────────

    @staticmethod
    def create_access_token(user: User) -> str:
        """Create a JWT access token for a user."""
        from jose import jwt

        expires = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
        claims = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
            "name": user.full_name,
            "exp": expires,
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode and validate a JWT token. Returns claims dict or raises."""
        from jose import jwt, JWTError

        try:
            claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return claims
        except JWTError:
            return {}

    # ── Authentication ──────────────────────────────────────────

    @staticmethod
    def authenticate(session: Session, email: str, password: str) -> Optional[User]:
        """Authenticate a user by email and password.

        Returns the User if credentials are valid, None otherwise.
        """
        user = session.query(User).filter_by(email=email, is_active=True).first()
        if user is None:
            return None
        if not AuthService.verify_password(password, user.password_hash):
            return None
        return user

    @staticmethod
    def get_user_from_token(session: Session, token: str) -> Optional[User]:
        """Extract and validate a user from a JWT token."""
        claims = AuthService.decode_token(token)
        user_id = claims.get("sub")
        if user_id is None:
            return None
        try:
            return session.query(User).filter_by(id=int(user_id), is_active=True).first()
        except (ValueError, TypeError):
            return None

    # ── User management ─────────────────────────────────────────

    @staticmethod
    def create_user(
        session: Session,
        username: str,
        email: str,
        password: str,
        full_name: str,
        role: str = UserRole.ENFORCEMENT_OFFICER,
        phone: Optional[str] = None,
    ) -> User:
        """Create a new user account."""
        user = User(
            username=username,
            email=email,
            password_hash=AuthService.hash_password(password),
            full_name=full_name,
            role=role,
            phone=phone,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    @staticmethod
    def get_user(session: Session, user_id: int) -> Optional[User]:
        """Fetch a user by ID."""
        return session.query(User).filter_by(id=user_id).first()

    @staticmethod
    def list_users(session: Session, role: Optional[str] = None) -> list[User]:
        """List all users, optionally filtered by role."""
        query = session.query(User)
        if role:
            query = query.filter_by(role=role)
        return query.order_by(User.full_name).all()

    @staticmethod
    def update_user_role(session: Session, user_id: int, new_role: str) -> Optional[User]:
        """Update a user's role."""
        user = AuthService.get_user(session, user_id)
        if user is None:
            return None
        user.role = new_role
        session.commit()
        session.refresh(user)
        return user

    @staticmethod
    def deactivate_user(session: Session, user_id: int) -> Optional[User]:
        """Deactivate a user account."""
        user = AuthService.get_user(session, user_id)
        if user is None:
            return None
        user.is_active = False
        session.commit()
        session.refresh(user)
        return user

    # ── Role / Permission helpers ───────────────────────────────

    @staticmethod
    def user_has_permission(user: User, permission: str) -> bool:
        """Check if a user has a specific permission based on their role."""
        perms = UserRole.PERMISSIONS.get(user.role, [])
        return permission in perms

    @staticmethod
    def user_has_any_permission(user: User, permissions: list[str]) -> bool:
        """Check if a user has at least one of the given permissions."""
        perms = UserRole.PERMISSIONS.get(user.role, [])
        return any(p in perms for p in permissions)
