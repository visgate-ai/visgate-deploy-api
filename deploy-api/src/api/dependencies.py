"""FastAPI dependencies: stateless auth, Firestore, rate limiting."""

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.cloud import firestore

from src.core.config import get_settings
from src.core.errors import RateLimitError, UnauthorizedError
from src.services.firestore_repo import get_firestore_client

security = HTTPBearer(auto_error=False)

# In-memory rate limit: subject -> list of request timestamps (sliding window)
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60.0  # seconds


@dataclass(frozen=True)
class RequestContext:
    runpod_api_key: str
    user_hash: str
    client_ip: str


def get_firestore() -> firestore.Client:
    """Return Firestore client for current project."""
    settings = get_settings()
    return get_firestore_client(settings.gcp_project_id)


def _check_rate_limit(subject: str, limit: int) -> None:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW
    store = _rate_limit_store[subject]
    store[:] = [t for t in store if t > window_start]
    if len(store) >= limit:
        raise RateLimitError(retry_after_seconds=60)
    store.append(now)


async def get_request_context(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)] = None,
    x_runpod_api_key: Annotated[Optional[str], Header(alias="X-Runpod-Api-Key")] = None,
) -> RequestContext:
    """Resolve stateless auth and rate-limit context using RUNPOD API key."""
    import os

    # DEV MODE: Skip auth for local testing
    if os.getenv("DEV_MODE") == "true":
        runpod_api_key = os.getenv("DEV_RUNPOD_KEY", "dev-runpod-key")
        user_hash = hashlib.sha256(runpod_api_key.encode("utf-8")).hexdigest()
        client_ip = request.client.host if request.client else "unknown"
        _check_rate_limit(user_hash, get_settings().rate_limit_requests_per_minute)
        return RequestContext(runpod_api_key=runpod_api_key, user_hash=user_hash, client_ip=client_ip)

    token: Optional[str] = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    if not token and x_runpod_api_key:
        token = x_runpod_api_key
    if not token:
        raise UnauthorizedError("Missing or invalid Runpod API key")

    user_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    client_ip = request.client.host if request.client else "unknown"
    settings = get_settings()

    # Rate limit per user_hash and IP (best-effort)
    _check_rate_limit(user_hash, settings.rate_limit_requests_per_minute)
    _check_rate_limit(client_ip, settings.rate_limit_requests_per_minute * 2)
    return RequestContext(runpod_api_key=token, user_hash=user_hash, client_ip=client_ip)


def verify_internal_webhook_secret(
    x_visgate_secret: Annotated[Optional[str], Header(alias="X-Visgate-Internal-Secret")] = None,
) -> None:
    """Optional: require header for internal deployment-ready endpoint."""
    settings = get_settings()
    if not settings.internal_webhook_secret:
        return
    if x_visgate_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")
