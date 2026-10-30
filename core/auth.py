"""Authentication routes and request dependencies."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Annotated, Any

import jwt
from jwt.exceptions import InvalidTokenError
import httpx
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token as google_id_token
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from redis.exceptions import RedisError
from sqlalchemy import and_, select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.context import current_user_id_ctx
from core.security import create_redis_client, get_allowed_origins
from storage.models import DBUser, DBExternalIdentity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])
security_scheme = HTTPBearer(auto_error=False)
password_hasher = PasswordHash.recommended()
REFRESH_COOKIE_MAX_AGE = 100*24*60*60
ACCESS_TOKEN_TTL = timedelta(minutes=30)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "http://localhost:5173/auth/callback",
).strip()


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).strip().lower()


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class GoogleCodeExchange(BaseModel):
    code: str = Field(min_length=1, max_length=2048)
    code_verifier: str = Field(
        min_length=43,
        max_length=128,
        pattern=r"^[A-Za-z0-9._~-]+$",
    )
    nonce: str = Field(min_length=32, max_length=128)    


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "").strip()
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET must be configured with at least 32 characters")
    return secret


def create_access_token(user_id: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
    {
        "sub": user_id, 
        "iat": now, 
        "exp": now + ACCESS_TOKEN_TTL,
        "jti": str(uuid.uuid4())
    },
        get_jwt_secret(),
        algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
    )


def issue_app_session(user: DBUser, response: Response) -> TokenResponse:
    user_id = str(user.id)
    refresh_token = secrets.token_urlsafe(48)
    store_refresh_token(refresh_token, user_id)
    _set_refresh_cookie(response, refresh_token)

    return TokenResponse(
        access_token=create_access_token(user_id=user_id),
        token_type="bearer",
    )


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hasher.verify(plain_password, hashed_password)


def _refresh_token_key(refresh_token: str) -> str:
    token_hash = sha256(refresh_token.encode("utf-8")).hexdigest()
    return f"lexory:refresh:{token_hash}"


def store_refresh_token(refresh_token: str, user_id: str) -> None:
    create_redis_client.set(
        _refresh_token_key(refresh_token), 
        user_id, 
        ex=REFRESH_COOKIE_MAX_AGE
    )


def _cookie_options() -> dict[str, Any]:
    secure = os.getenv("AUTH_COOKIE_SECURE", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    return {
        "httponly": True,
        "secure": secure,
        "samesite": "none" if secure else "lax",
        "path": "/",
    }


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="refresh_token",
        value=token,
        max_age=REFRESH_COOKIE_MAX_AGE,
        **_cookie_options(),
    )


def _delete_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key="refresh_token", **_cookie_options())


def _validate_browser_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    same_origin = str(request.base_url).rstrip("/")
    if origin and origin not in get_allowed_origins() and origin != same_origin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin is not allowed",
        )


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(security_scheme)
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AsyncIterator[DBUser]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise credentials_exception

    try:
        payload = jwt.decode(
            credentials.credentials,
            get_jwt_secret(),
            algorithms=[os.getenv("JWT_ALGORITHM", "HS256")],
            options={"require": ["exp", "sub"]},
        )
    except InvalidTokenError:
        raise credentials_exception from None

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise credentials_exception

    context_token = current_user_id_ctx.set(user_id)
    try:
        user = await db.get(DBUser, user_id)
        if user is None:
            raise credentials_exception
        yield user
    finally:
        current_user_id_ctx.reset(context_token)


@router.post("/register", response_model=UserResponse, status_code=201)
async def register_user(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DBUser:
    existing = await db.scalar(select(DBUser).where(DBUser.email == user_data.email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered",
        )

    user = DBUser(
        id=str(uuid.uuid4()),
        email=str(user_data.email),
        hashed_password=await asyncio.to_thread(hash_password, user_data.password),
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered",
        ) from exc
    await db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
async def login_user(
    user_data: UserCreate,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    _validate_browser_origin(request)
    user = await db.scalar(select(DBUser).where(DBUser.email == user_data.email))
    if user is None or not await asyncio.to_thread(
        verify_password, user_data.password, user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    refresh_token = secrets.token_urlsafe(48)
    try:
        await request.app.state.redis_client.set(
            _refresh_token_key(refresh_token),
            user.id,
        )
    except RedisError as exc:
        logger.exception("Could not create an authentication session in Redis")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication session store is unavailable",
        ) from exc

    _set_refresh_cookie(response, refresh_token)
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/google/config")
def get_google_auth_config() -> dict[str, str]:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured",
        )
    return {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
    }


def exchange_google_code(code: str, code_verifier: str) -> str:
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured",
        )

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                    "code_verifier": code_verifier,
                },
            )
            response.raise_for_status()
            token_response = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("Google authorization-code exchange failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed",
        ) from exc

    id_token = token_response.get("id_token") if isinstance(token_response, dict) else None
    if not isinstance(id_token, str) or not id_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed",
        )
    return id_token


def verify_google_identity(token: str, expected_nonce: str) -> tuple[str, str]:
    try:
        claims = google_id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            GOOGLE_CLIENT_ID,
        )
    except (ValueError, GoogleAuthError) as exc:
        logger.info("Google ID-token validation failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed",
        ) from exc

    if claims.get("nonce") != expected_nonce:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed",
        )

    subject = claims.get("sub")
    email = claims.get("email")
    if (
        not isinstance(subject, str)
        or not subject
        or not isinstance(email, str)
        or not email
        or claims.get("email_verified") is not True
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google authentication failed",
        )

    return subject, email.strip().lower()


@router.post("/google/exchange", response_model=TokenResponse)
async def google_code_exchange(
    payload: GoogleCodeExchange,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    _validate_browser_origin(request)

    id_token = await asyncio.to_thread(
        exchange_google_code, payload.code, payload.code_verifier
    )
    provider_subject, email = await asyncio.to_thread(
        verify_google_identity, id_token, payload.nonce
    )

    identity_query = select(DBExternalIdentity).where(
        and_(
            DBExternalIdentity.provider == "google",
            DBExternalIdentity.provider_subject == provider_subject,
        )
    )

    result = await db.execute(identity_query)
    identity = result.scalars().first()

    if identity is not None:
        user = await db.get(DBUser, identity.user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Google authentication failed",
            )
        return await issue_app_session(user, response, request.app.state.redis_client)

    user = DBUser(email=email, hashed_password=None)
    identity = DBExternalIdentity(
        provider="google",
        provider_subject=provider_subject,
        user=user,
    )
    db.add(user)
    db.add(identity)
    try:
        await db.commit()
        await db.refresh(user)
    except IntegrityError as exc:
        await db.rollback()
        result = await db.execute(identity_query)
        identity = result.scalars().first()
        if identity is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Could not create Google account",
            ) from exc
        user = await db.get(DBUser, identity.user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Google authentication failed",
            ) from exc

    return await issue_app_session(user, response, request.app.state.redis_client)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_session(request: Request, response: Response) -> TokenResponse:
    _validate_browser_origin(request)
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    try:
        user_id = await request.app.state.redis_client.getdel(
            _refresh_token_key(refresh_token)
        )
    except RedisError as exc:
        logger.exception("Could not refresh an authentication session in Redis")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication session store is unavailable",
        ) from exc

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    new_refresh_token = secrets.token_urlsafe(48)
    try:
        await request.app.state.redis_client.set(
            _refresh_token_key(new_refresh_token),
            user_id,
        )
    except RedisError as exc:
        logger.exception("Could not rotate an authentication session in Redis")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication session store is unavailable",
        ) from exc

    _set_refresh_cookie(response, new_refresh_token)
    return TokenResponse(access_token=create_access_token(user_id))


@router.post("/logout")
async def logout(request: Request, response: Response) -> dict[str, str]:
    _validate_browser_origin(request)
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        try:
            await request.app.state.redis_client.delete(
                _refresh_token_key(refresh_token)
            )
        except RedisError as exc:
            logger.exception("Could not revoke an authentication session in Redis")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication session store is unavailable",
            ) from exc

    _delete_refresh_cookie(response)
    return {"detail": "Logged out"}
