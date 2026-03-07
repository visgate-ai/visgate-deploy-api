"""Helpers for inference job state, IDs, payload compaction, and output metadata."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def generate_job_id() -> str:
    return f"job_{datetime.now(UTC).strftime('%Y%m%d')}_{secrets.token_hex(4)}"


def map_provider_status(status: str | None) -> str:
    normalized = (status or "").upper()
    if normalized in {"IN_QUEUE"}:
        return "queued"
    if normalized in {"IN_PROGRESS", "RUNNING"}:
        return "running"
    if normalized in {"COMPLETED", "READY"}:
        return "completed"
    if normalized in {"CANCELLED"}:
        return "cancelled"
    if normalized in {"FAILED", "TIMED_OUT"}:
        return "failed"
    if normalized in {"EXPIRED"}:
        return "expired"
    return "running"


def compact_payload(value: Any, *, max_string: int = 512, max_items: int = 12) -> Any:
    """Compact large provider outputs before persisting them in Firestore."""
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return {"truncated": True, "length": len(value), "preview": value[:max_string]}
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [compact_payload(item, max_string=max_string, max_items=max_items) for item in value[:max_items]]
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                compacted["__truncated_keys__"] = len(value) - max_items
                break
            if key.lower().endswith("_base64"):
                raw_length = len(item) if isinstance(item, str) else None
                compacted[key] = {"stored": False, "reason": "base64_omitted", "length": raw_length}
                continue
            compacted[key] = compact_payload(item, max_string=max_string, max_items=max_items)
        return compacted
    return str(value)


def sanitize_s3_config(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    return {
        "bucket_name": value.get("bucketName"),
        "endpoint_url": value.get("endpointUrl"),
        "key_prefix": value.get("keyPrefix"),
    }


def _find_first_url(value: Any) -> str | None:
    if isinstance(value, str):
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https", "s3"} and parsed.netloc:
            return value
        return None
    if isinstance(value, list):
        for item in value:
            found = _find_first_url(item)
            if found:
                return found
        return None
    if isinstance(value, dict):
        preferred_keys = (
            "url",
            "image_url",
            "audio_url",
            "video_url",
            "artifact_url",
            "output_url",
            "s3_url",
        )
        for key in preferred_keys:
            if key in value:
                found = _find_first_url(value.get(key))
                if found:
                    return found
        for item in value.values():
            found = _find_first_url(item)
            if found:
                return found
    return None


def _extract_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key_name in ("key", "object_key", "path", "filename"):
            key_value = value.get(key_name)
            if isinstance(key_value, str) and key_value:
                return key_value
        for item in value.values():
            found = _extract_key(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _extract_key(item)
            if found:
                return found
    return None


def extract_artifact_metadata(output: Any, destination: dict[str, Any] | None) -> dict[str, Any] | None:
    if output is None and not destination:
        return None
    artifact: dict[str, Any] = {
        "bucket_name": destination.get("bucket_name") if destination else None,
        "endpoint_url": destination.get("endpoint_url") if destination else None,
        "key": None,
        "url": None,
        "content_type": None,
        "bytes": None,
    }
    if isinstance(output, dict):
        artifact["content_type"] = output.get("content_type") or output.get("mime_type")
        artifact["bytes"] = output.get("bytes") or output.get("size") or output.get("size_bytes")
    artifact["url"] = _find_first_url(output)
    artifact["key"] = _extract_key(output)
    if artifact["url"] and not artifact["key"]:
        parsed = urlparse(artifact["url"])
        artifact["key"] = parsed.path.lstrip("/") or None
    if not any(artifact.values()):
        return None
    return artifact


def build_job_metrics(
    *,
    created_at: str | None,
    completed_at: str | None,
    queue_ms: int | None = None,
    execution_ms: int | None = None,
) -> dict[str, Any]:
    wall_clock_ms = None
    created_dt = parse_iso(created_at)
    completed_dt = parse_iso(completed_at)
    if created_dt and completed_dt:
        wall_clock_ms = max(int((completed_dt - created_dt).total_seconds() * 1000), 0)
    return {
        "queue_ms": queue_ms,
        "execution_ms": execution_ms,
        "wall_clock_ms": wall_clock_ms,
    }


def extract_estimated_cost(raw_payload: dict[str, Any] | None) -> float | None:
    if not raw_payload:
        return None
    for key in ("costUSD", "costUsd", "estimatedCostUsd", "estimated_cost_usd", "cost"):
        value = raw_payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def resolve_gpu_hourly_price(gpu_allocated: str | None, gpu_types: list[dict[str, Any]]) -> float | None:
    if not gpu_allocated:
        return None
    alloc = gpu_allocated.lower()
    for gpu in gpu_types:
        name = str(gpu.get("displayName", "")).lower()
        gpu_id = str(gpu.get("id", "")).lower()
        if alloc in name or name in alloc or alloc == gpu_id:
            price = gpu.get("communityPrice") or gpu.get("securePrice")
            if isinstance(price, (int, float)):
                return float(price)
    return None


def estimate_cost_from_execution(execution_ms: int | None, price_per_hour: float | None) -> float | None:
    if execution_ms is None or price_per_hour is None:
        return None
    hours = max(execution_ms / 1000 / 3600, 0.0)
    return round(hours * price_per_hour, 6)