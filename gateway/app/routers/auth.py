"""Authentication routes (mounted under prefix /api/auth in main.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import authenticate_user, create_access_token, get_current_user
from app.config import get_settings
from app.db import get_db
from app.models import User
from app.schemas import LoginRequest, MeResponse, TokenResponse

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = authenticate_user(db, body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials"
        )
    token = create_access_token(sub=user.username, extra={"role": user.role})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=get_settings().jwt_expires_in,
    )


@router.get("/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)) -> MeResponse:
    return MeResponse(id=user.id, username=user.username, role=user.role)
