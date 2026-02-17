"""Ephemeral in-memory secret cache (stateless mode)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional


@dataclass
class CachedSecrets:
    runpod_api_key: str
    hf_token: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_endpoint_url: Optional[str]
    s3_model_url: Optional[str]
    expires_at: float


_DEFAULT_TTL_SECONDS = 3600.0
_cache: dict[str, CachedSecrets] = {}
_lock = Lock()


def store_secrets(
    deployment_id: str,
    runpod_api_key: str,
    hf_token: Optional[str],
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_endpoint_url: Optional[str] = None,
    s3_model_url: Optional[str] = None,
    ttl_seconds: float = _DEFAULT_TTL_SECONDS,
) -> None:
    """Store secrets in memory for best-effort background tasks."""
    expires_at = time.monotonic() + ttl_seconds
    with _lock:
        _cache[deployment_id] = CachedSecrets(
            runpod_api_key=runpod_api_key,
            hf_token=hf_token,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_endpoint_url=aws_endpoint_url,
            s3_model_url=s3_model_url,
            expires_at=expires_at,
        )


def get_secrets(deployment_id: str) -> Optional[CachedSecrets]:
    """Fetch secrets if present and not expired."""
    now = time.monotonic()
    with _lock:
        entry = _cache.get(deployment_id)
        if not entry:
            return None
        if entry.expires_at <= now:
            _cache.pop(deployment_id, None)
            return None
        return entry


def clear_secrets(deployment_id: str) -> None:
    """Remove secrets after use."""
    with _lock:
        _cache.pop(deployment_id, None)
