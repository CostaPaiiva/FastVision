from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.security import (
    TokenPair,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterIn(BaseModel):
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class RefreshIn(BaseModel):
    refresh_token: str


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenOut)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 chars")

    existing = db.get(User, email)
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    user = User(email=email, password_hash=hash_password(payload.password), role="user")
    db.add(user)
    db.commit()

    access = create_access_token(subject=email, role=user.role)
    refresh = create_refresh_token(subject=email, role=user.role)
    return TokenOut(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.get(User, email)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access = create_access_token(subject=email, role=user.role)
    refresh = create_refresh_token(subject=email, role=user.role)
    return TokenOut(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenOut)
def refresh(payload: RefreshIn, db: Session = Depends(get_db)):
    token = payload.refresh_token
    try:
        data = decode_token(token)
        if data.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        email = (data.get("sub") or "").lower().strip()
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token subject")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.get(User, email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access = create_access_token(subject=email, role=user.role)
    refresh_t = create_refresh_token(subject=email, role=user.role)
    return TokenOut(access_token=access, refresh_token=refresh_t)
