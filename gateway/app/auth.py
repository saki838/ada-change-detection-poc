"""Auth: bcrypt password hashing, HS256 JWT issue/decode, and FastAPI deps."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# tokenUrl points at the login route; only used by the OpenAPI docs "Authorize".
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except ValueError:
        return False


def create_access_token(sub: str, extra: dict | None = None) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: dict = {
        "sub": sub,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.jwt_expires_in)).timestamp()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


def decode_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    except JWTError:
        raise _UNAUTHENTICATED


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not token:
        raise _UNAUTHENTICATED
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise _UNAUTHENTICATED
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        raise _UNAUTHENTICATED
    return user


def ensure_seed_admin(db: Session) -> None:
    """Idempotently create the demo admin (admin/admin123) if absent.

    The compose stack normally seeds this via db/init/02_seed.sql; this startup
    hook guarantees the demo credential exists even if that seed is missing.
    """
    existing = db.scalar(select(User).where(User.username == "admin"))
    if existing is not None:
        return
    db.add(
        User(
            username="admin",
            password_hash=hash_password("admin123"),
            role="admin",
        )
    )
    db.commit()
