"""FastAPI dependencies: auth, Firestore, rate limiting."""

import time
from collections import defaultdict
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.cloud import firestore

from src.core.config import get_settings
from src.core.errors import RateLimitError, UnauthorizedError
from src.services.firestore_repo import get_api_key, get_firestore_client

security = HTTPBearer(auto_error=False)

# In-memory rate limit: key_id -> list of request timestamps (sliding window)
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60.0  # seconds


def get_firestore() -> firestore.Client:
    """Return Firestore client for current project."""
    settings = get_settings()
    return get_firestore_client(settings.gcp_project_id)


def _check_rate_limit(key_id: str, limit: int) -> None:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW
    store = _rate_limit_store[key_id]
    store[:] = [t for t in store if t > window_start]
    if len(store) >= limit:
        raise RateLimitError(retry_after_seconds=60)
    store.append(now)


async def get_current_api_key(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)] = None,
    firestore_client: firestore.Client = Depends(get_firestore),
) -> str:
    """
    Validate Bearer token against Firestore api_keys collection.
    Key must exist and be active.
    """
    import os
    # DEV MODE: Skip auth for local testing
    if os.getenv("DEV_MODE") == "true":
        return "dev-api-key"
    
    token: Optional[str] = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    if not token:
        raise UnauthorizedError("Missing or invalid API key")
    
    settings = get_settings()
    key_doc = get_api_key(firestore_client, settings.firestore_collection_api_keys, token)
    
    if not key_doc:
        raise UnauthorizedError("Invalid API key")
    
    if not key_doc.get("active", True):
        raise UnauthorizedError("API key is revoked or inactive")

    # Use key ID (the token itself) or a user ID from the doc
    key_id = token 
    
    _check_rate_limit(key_id, settings.rate_limit_requests_per_minute)
    return key_id


def verify_internal_webhook_secret(
    x_visgate_secret: Annotated[Optional[str], Header(alias="X-Visgate-Internal-Secret")] = None,
) -> None:
    """Optional: require header for internal deployment-ready endpoint."""
    settings = get_settings()
    if not settings.internal_webhook_secret:
        return
    if x_visgate_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")
