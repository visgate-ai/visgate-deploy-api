"""Cloudflare R2 model cache manifest reader/writer.

Reads and writes ``models/manifest.json`` from the configured R2 bucket.
Falls back gracefully whenever R2 is not configured or unreachable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

from src.core.config import get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MANIFEST_KEY = "models/manifest.json"
_CACHE_LOCK = threading.Lock()
_MANIFEST_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def _cache_key(endpoint_url: str, bucket: str) -> tuple[str, str]:
    return (endpoint_url.rstrip("/"), bucket)


def _get_cached_ids(endpoint_url: str, bucket: str) -> set[str] | None:
    key = _cache_key(endpoint_url, bucket)
    with _CACHE_LOCK:
        entry = _MANIFEST_CACHE.get(key)
        if not entry:
            return None
        expires_at = float(entry.get("expires_at", 0.0))
        if expires_at < time.monotonic():
            return None
        return set(entry.get("model_ids", set()))


def _update_cached_ids(
    endpoint_url: str,
    bucket: str,
    model_ids: set[str],
    ttl_seconds: int | None = None,
) -> None:
    ttl = get_settings().cache_manifest_ttl_seconds if ttl_seconds is None else ttl_seconds
    key = _cache_key(endpoint_url, bucket)
    with _CACHE_LOCK:
        _MANIFEST_CACHE[key] = {
            "model_ids": set(model_ids),
            "expires_at": time.monotonic() + max(ttl, 0),
        }


def invalidate_cached_model_ids(
    endpoint_url: str,
    bucket: str = "visgate-models",
    model_id: str | None = None,
) -> None:
    """Invalidate or locally update the in-memory manifest cache."""
    key = _cache_key(endpoint_url, bucket)
    with _CACHE_LOCK:
        if model_id is None:
            _MANIFEST_CACHE.pop(key, None)
            return
        entry = _MANIFEST_CACHE.get(key)
        if not entry:
            return
        model_ids = set(entry.get("model_ids", set()))
        model_ids.add(model_id)
        entry["model_ids"] = model_ids
        entry["expires_at"] = time.monotonic() + max(get_settings().cache_manifest_ttl_seconds, 0)


def _create_s3_client(endpoint_url: str, access_key_id: str, secret_access_key: str):
    import boto3  # type: ignore

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    )


def _read_manifest(s3, bucket: str) -> dict:
    from botocore.exceptions import ClientError  # type: ignore

    try:
        response = s3.get_object(Bucket=bucket, Key=_MANIFEST_KEY)
        manifest: dict = json.loads(response["Body"].read())
        return manifest
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404", "NotFound"}:
            return {"models": []}
        raise


def fetch_cached_model_ids(
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str = "visgate-models",
    force_refresh: bool = False,
) -> set[str]:
    """Return the set of model IDs present in the R2 manifest.

    Returns an empty set on any error so failures are non-fatal.
    """
    cached_ids = None if force_refresh else _get_cached_ids(endpoint_url, bucket)
    if cached_ids is not None:
        return cached_ids

    try:
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore

        s3 = _create_s3_client(endpoint_url, access_key_id, secret_access_key)
        manifest = _read_manifest(s3, bucket)
        model_ids: list[str] = manifest.get("models", [])
        cached = set(model_ids)
        _update_cached_ids(endpoint_url, bucket, cached)
        logger.debug("R2 manifest loaded — %d cached models", len(model_ids))
        return cached
    except ImportError:
        logger.warning("boto3 not installed; R2 manifest unavailable")
        return set()
    except (BotoCoreError, ClientError) as exc:
        logger.warning("R2 manifest read failed: %s", exc)
        stale_ids = _get_cached_ids(endpoint_url, bucket)
        if stale_ids is not None:
            logger.warning("Using stale in-memory manifest after read failure")
            return stale_ids
        return set()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error reading R2 manifest: %s", exc)
        stale_ids = _get_cached_ids(endpoint_url, bucket)
        if stale_ids is not None:
            logger.warning("Using stale in-memory manifest after unexpected read failure")
            return stale_ids
        return set()


def model_s3_url(base_url: str, model_id: str) -> str:
    """Return the per-model S3 URL for a given HuggingFace model ID.

    Example: base_url="s3://visgate-models/models", model_id="black-forest-labs/FLUX.1-dev"
    → "s3://visgate-models/models/black-forest-labs--FLUX.1-dev"
    """
    slug = model_id.replace("/", "--")
    return f"{base_url.rstrip('/')}/{slug}"


def add_model_to_manifest(
    model_id: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str = "visgate-models",
    max_attempts: int = 3,
) -> bool:
    """Add a model ID to the R2 manifest JSON. Returns True on success."""
    try:
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore

        s3 = _create_s3_client(endpoint_url, access_key_id, secret_access_key)
        for attempt in range(1, max_attempts + 1):
            manifest = _read_manifest(s3, bucket)
            models = set(manifest.get("models", []))
            if model_id in models:
                invalidate_cached_model_ids(endpoint_url, bucket, model_id=model_id)
                return True

            models.add(model_id)
            manifest["models"] = sorted(models)
            s3.put_object(
                Bucket=bucket,
                Key=_MANIFEST_KEY,
                Body=json.dumps(manifest, indent=2).encode("utf-8"),
                ContentType="application/json",
            )

            refreshed = fetch_cached_model_ids(
                endpoint_url=endpoint_url,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                bucket=bucket,
                force_refresh=True,
            )
            if model_id in refreshed:
                invalidate_cached_model_ids(endpoint_url, bucket, model_id=model_id)
                logger.info("R2 manifest updated — added %s", model_id)
                return True

            logger.warning(
                "R2 manifest verification failed for %s on attempt %d/%d",
                model_id,
                attempt,
                max_attempts,
            )
            time.sleep(min(0.25 * attempt, 1.0))
        return False
    except ImportError:
        logger.warning("boto3 not installed; cannot update R2 manifest")
        return False
    except (BotoCoreError, ClientError) as exc:  # type: ignore[name-defined]
        logger.error("R2 manifest write failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error updating R2 manifest: %s", exc)
        return False
