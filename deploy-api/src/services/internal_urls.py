"""Helpers for building internal callback URLs."""

from __future__ import annotations

from fastapi import Request

from src.core.config import get_settings


def resolve_internal_base_url(request: Request | None = None) -> str:
    """Return an absolute base URL for internal callbacks."""
    configured = (get_settings().internal_webhook_base_url or "").strip().rstrip("/")
    if configured:
        return configured
    if request is None:
        return ""

    forwarded_proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "https").split(",", 1)[0].strip()
    forwarded_host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    ).split(",", 1)[0].strip()
    root_path = (request.scope.get("root_path") or "").rstrip("/")

    if forwarded_host:
        return f"{forwarded_proto}://{forwarded_host}{root_path}".rstrip("/")
    return str(request.base_url).rstrip("/")


def build_deployment_ready_url(base_url: str, deployment_id: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base:
        return ""
    return f"{base}/internal/deployment-ready/{deployment_id}"


def build_log_tunnel_url(base_url: str, deployment_id: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base:
        return ""
    return f"{base}/internal/logs/{deployment_id}"


def build_inference_job_callback_url(base_url: str, job_id: str, secret: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base or not secret:
        return ""
    return f"{base}/internal/inference/jobs/{job_id}/complete?secret={secret}"