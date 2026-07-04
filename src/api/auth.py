"""FastAPI router for authentication and user management.

Endpoints:
    POST /api/v1/auth/login       — Authenticate and receive JWT token
    GET  /api/v1/auth/me          — Get current user profile
    POST /api/v1/auth/users       — Create new user (admin only)
    GET  /api/v1/auth/users       — List all users (admin/supervisor)
    PATCH /api/v1/auth/users/{id}/role — Update user role (admin only)
    POST /api/v1/auth/users/{id}/deactivate — Deactivate user (admin only)
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src.database.connection import get_engine
from src.database.models import init_db, UserRole
from src.services.auth_service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

DB_PATH = os.getenv("ADA_DB_PATH", "data/ada.db")
init_db(DB_PATH)


# ── Pydantic schemas ────────────────────────────────────────────────


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    role: str = UserRole.ENFORCEMENT_OFFICER
    phone: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    phone: Optional[str]
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    total: int
    users: list[UserResponse]


class RoleUpdateRequest(BaseModel):
    role: str


# ── Helpers ─────────────────────────────────────────────────────────


def _get_session():
    engine = get_engine(DB_PATH)
    return Session(engine)


def _user_to_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        phone=user.phone,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else "",
    )


def _get_token_from_header(authorization: str = "") -> str:
    """Extract Bearer token from Authorization header."""
    if not authorization.startswith("Bearer "):
        return ""
    return authorization[7:]


def _require_auth(authorization: str = Header("")):
    """Dependency: require a valid JWT token. Returns the authenticated User.

    Reads the Authorization HTTP header (e.g. "Bearer <token>") and validates
    the JWT.  FastAPI automatically maps the snake_case parameter name
    `authorization` to the HTTP header `Authorization`.
    """
    token = _get_token_from_header(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    with _get_session() as session:
        user = AuthService.get_user_from_token(session, token)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user


def _require_role(*roles: str):
    """Dependency factory: require the user to have one of the specified roles."""
    def role_checker(user=Depends(_require_auth)):
        if user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Requires one of these roles: {', '.join(roles)}",
            )
        return user
    return role_checker


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/login", response_model=LoginResponse, summary="Authenticate and get JWT token")
def login(body: LoginRequest):
    """Login with email and password. Returns a JWT access token."""
    with _get_session() as session:
        user = AuthService.authenticate(session, body.email, body.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = AuthService.create_access_token(user)
    return LoginResponse(
        access_token=token,
        user=_user_to_response(user),
    )


@router.get("/me", response_model=UserResponse, summary="Get current user profile")
def get_me(user=Depends(_require_auth)):
    """Return the profile of the currently authenticated user."""
    return _user_to_response(user)


@router.post(
    "/users",
    response_model=UserResponse,
    summary="Create a new user (admin only)",
)
def create_user(
    body: UserCreateRequest,
    admin=Depends(_require_role(UserRole.ADMIN)),
):
    """Create a new user account. Only accessible by admins."""
    with _get_session() as session:
        # Check for duplicates
        if session.query(type(admin)).filter_by(email=body.email).first():
            raise HTTPException(status_code=409, detail="Email already registered")
        if session.query(type(admin)).filter_by(username=body.username).first():
            raise HTTPException(status_code=409, detail="Username already taken")

        user = AuthService.create_user(
            session,
            username=body.username,
            email=body.email,
            password=body.password,
            full_name=body.full_name,
            role=body.role,
            phone=body.phone,
        )
    return _user_to_response(user)


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List all users (admin/supervisor)",
)
def list_users(
    role: Optional[str] = Query(None, description="Filter by role"),
    user=Depends(_require_role(UserRole.ADMIN, UserRole.SUPERVISOR)),
):
    """List all users, optionally filtered by role."""
    with _get_session() as session:
        users = AuthService.list_users(session, role=role)
    return UserListResponse(total=len(users), users=[_user_to_response(u) for u in users])


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
    summary="Update user role (admin only)",
)
def update_user_role(
    user_id: int,
    body: RoleUpdateRequest,
    admin=Depends(_require_role(UserRole.ADMIN)),
):
    """Change a user's role. Only accessible by admins."""
    with _get_session() as session:
        user = AuthService.update_user_role(session, user_id, body.role)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return _user_to_response(user)


@router.post(
    "/users/{user_id}/deactivate",
    response_model=UserResponse,
    summary="Deactivate a user (admin only)",
)
def deactivate_user(
    user_id: int,
    admin=Depends(_require_role(UserRole.ADMIN)),
):
    """Deactivate a user account. Only accessible by admins."""
    with _get_session() as session:
        user = AuthService.deactivate_user(session, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return _user_to_response(user)
