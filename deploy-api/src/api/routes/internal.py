"""Internal routes: deployment-ready callback from inference container."""

import asyncio
import fnmatch
import json
import os
import time
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.logging import structured_log
from src.models.schemas import DeploymentReadyPayload
from src.services.deployment import (
    mark_deployment_ready_and_notify,
    orchestrate_deployment,
    update_deployment_phase_from_worker,
)
from src.services.firestore_repo import get_deployment, update_deployment
from src.services.log_tunnel import append_live_log
from src.services.provider_factory import get_provider
from src.services.r2_manifest import add_model_to_manifest, fetch_cached_model_ids, model_s3_url
from src.services.secret_cache import get_secrets

router = APIRouter(prefix="/internal", tags=["internal"])


class LiveLogPayload(BaseModel):
    level: str = "INFO"
    message: str


class CleanupPayload(BaseModel):
    reason: str = "idle_timeout"
    runpod_api_key: str | None = None  # Passed by worker; eliminates multi-instance secret_cache dependency


class ModelCachedPayload(BaseModel):
    """Payload for worker → /internal/model-cached callback."""

    hf_model_id: str
    deployment_id: str


class CacheModelPayload(BaseModel):
    """Payload for Cloud Tasks → /internal/tasks/cache-model."""

    hf_model_id: str
    hf_token: str = ""


_CACHE_IGNORE_PATTERNS = ["*.msgpack", "*.h5", "flax_model*", "rust_model.ot", "tf_model*"]


def _task_log(level: str, message: str, **metadata: object) -> None:
    structured_log(level, message, operation="r2.cache_model", metadata=metadata)


def _should_skip_file(path: str) -> bool:
    """Return True if this file matches _CACHE_IGNORE_PATTERNS."""
    return any(fnmatch.fnmatch(os.path.basename(path), pat) for pat in _CACHE_IGNORE_PATTERNS)


def _cache_model_once(hf_model_id: str, hf_token: str | None) -> dict[str, object]:
    """Stream model files from HuggingFace directly to R2 — no local disk required."""
    import boto3
    import requests
    from boto3.s3.transfer import TransferConfig
    from huggingface_hub import list_repo_tree

    settings = get_settings()

    cached_ids = fetch_cached_model_ids(
        endpoint_url=settings.r2_endpoint_url,
        access_key_id=settings.r2_access_key_id_rw,
        secret_access_key=settings.r2_secret_access_key_rw,
        force_refresh=True,
    )
    if hf_model_id in cached_ids:
        return {"status": "already_cached", "files_uploaded": 0, "manifest_updated": True}

    model_slug = hf_model_id.replace("/", "--")
    s3_prefix = f"models/{model_slug}"
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id_rw,
        aws_secret_access_key=settings.r2_secret_access_key_rw,
        region_name="auto",
    )

    # Force multipart for files > 64 MB so R2 never sees an unbounded single-PUT.
    # Without this, boto3 can't infer Content-Length from a streaming response and
    # may produce a malformed request on large model shards (e.g. FLUX 12 GB).
    transfer_cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,  # 64 MB
        multipart_chunksize=64 * 1024 * 1024,  # 64 MB per part
        use_threads=False,                       # single-threaded inside Cloud Tasks
    )

    hf_headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    file_count = 0
    for item in list_repo_tree(hf_model_id, recursive=True, token=hf_token or None):
        if getattr(item, "type", None) != "file":
            continue
        file_path = item.path
        if _should_skip_file(file_path):
            continue

        url = f"https://huggingface.co/{hf_model_id}/resolve/main/{file_path}"
        resp = requests.get(url, stream=True, headers=hf_headers, timeout=600)
        resp.raise_for_status()
        resp.raw.decode_content = True  # handle gzip/deflate transparently

        s3_key = f"{s3_prefix}/{file_path}"
        file_size = int(resp.headers.get("Content-Length", 0)) or None
        extra_args = {"ContentLength": file_size} if file_size else {}
        s3.upload_fileobj(resp.raw, "visgate-models", s3_key, ExtraArgs=extra_args, Config=transfer_cfg)
        file_count += 1
        _task_log("INFO", f"Streamed to R2: {file_path}", s3_key=s3_key, size_bytes=file_size)

    manifest_updated = add_model_to_manifest(
        model_id=hf_model_id,
        endpoint_url=settings.r2_endpoint_url,
        access_key_id=settings.r2_access_key_id_rw,
        secret_access_key=settings.r2_secret_access_key_rw,
    )
    if not manifest_updated:
        raise RuntimeError("manifest update failed after model upload")
    return {"status": "ok", "files_uploaded": file_count, "manifest_updated": True}


class OrchestrateTaskPayload(BaseModel):
    """Payload for Cloud Tasks → /internal/tasks/orchestrate-deployment."""

    deployment_id: str
    secret_ref: str  # Secret Manager secret name; fetched and version-destroyed on first use


def _fetch_and_destroy_task_secrets(secret_ref: str, project_id: str) -> dict:
    """
    Fetch per-deployment credentials from Secret Manager and immediately destroy the
    secret version so credentials are not left at rest after orchestration starts.
    """
    from google.cloud import secretmanager  # lazy import

    client = secretmanager.SecretManagerServiceClient()
    version_path = f"projects/{project_id}/secrets/{secret_ref}/versions/latest"
    response = client.access_secret_version(request={"name": version_path})
    data = json.loads(response.payload.data.decode())

    try:
        client.destroy_secret_version(request={"name": version_path})
    except Exception:
        pass  # Non-critical; the secret version will eventually expire

    return data


@router.post("/tasks/orchestrate-deployment")
async def run_orchestration_task(
    payload: OrchestrateTaskPayload,
) -> dict:
    """
    Cloud Tasks target: receives deployment_id + secret_ref, fetches credentials from
    Secret Manager, destroys the secret version, then kicks off orchestrate_deployment.

    This endpoint is called by Google Cloud Tasks only (authenticated via OIDC token).
    No additional secret header check needed — OIDC provides sufficient authentication.
    """
    settings = get_settings()

    try:
        secrets = _fetch_and_destroy_task_secrets(payload.secret_ref, settings.gcp_project_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch task secrets: {exc}")

    asyncio.create_task(
        orchestrate_deployment(
            payload.deployment_id,
            secrets["runpod_api_key"],
            secrets.get("hf_token"),
            secrets.get("aws_access_key_id"),
            secrets.get("aws_secret_access_key"),
            secrets.get("aws_endpoint_url"),
            secrets.get("s3_model_url"),
        )
    )
    return {"status": "accepted", "deployment_id": payload.deployment_id}


@router.post("/tasks/cache-model")
async def task_cache_model(
    payload: CacheModelPayload,
    x_visgate_secret: Annotated[
        str | None,
        Header(alias="X-Visgate-Internal-Secret"),
    ] = None,
    x_cloudtasks_taskname: Annotated[
        str | None,
        Header(alias="X-CloudTasks-TaskName"),
    ] = None,
    x_cloudtasks_taskretrycount: Annotated[
        str | None,
        Header(alias="X-CloudTasks-TaskRetryCount"),
    ] = None,
) -> dict:
    """
    Cloud Tasks target: download HF model to temp dir, upload all files to R2, update manifest.
    Runs independently as a background job.
    """
    settings = get_settings()

    if settings.internal_webhook_secret and x_visgate_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")

    if not settings.r2_access_key_id_rw or not settings.r2_endpoint_url:
        return {"status": "skipped", "reason": "R2 not configured"}

    hf_model_id = payload.hf_model_id
    hf_token = payload.hf_token or None
    task_name = x_cloudtasks_taskname or "manual"
    retry_count = int(x_cloudtasks_taskretrycount or "0")
    started_at = time.perf_counter()
    r2_path = model_s3_url(settings.r2_model_base_url, hf_model_id)

    _task_log(
        "INFO",
        "Cache-model task started",
        hf_model_id=hf_model_id,
        task_name=task_name,
        retry_count=retry_count,
        r2_path=r2_path,
    )

    last_error: Exception | None = None
    for attempt in range(1, settings.cache_model_task_max_retries + 1):
        attempt_started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_cache_model_once, hf_model_id, hf_token),
                timeout=settings.cache_model_task_timeout_seconds,
            )
            duration_ms = (time.perf_counter() - started_at) * 1000
            _task_log(
                "INFO",
                "Cache-model task completed",
                hf_model_id=hf_model_id,
                task_name=task_name,
                retry_count=retry_count,
                attempt=attempt,
                files_uploaded=result.get("files_uploaded", 0),
                status=result.get("status", "ok"),
                duration_ms=round(duration_ms, 2),
            )
            return {
                "status": result.get("status", "ok"),
                "files_uploaded": result.get("files_uploaded", 0),
                "manifest_updated": result.get("manifest_updated", True),
                "task_name": task_name,
                "attempt": attempt,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempt_duration_ms = (time.perf_counter() - attempt_started) * 1000
            _task_log(
                "WARNING",
                "Cache-model attempt failed",
                hf_model_id=hf_model_id,
                task_name=task_name,
                retry_count=retry_count,
                attempt=attempt,
                max_attempts=settings.cache_model_task_max_retries,
                duration_ms=round(attempt_duration_ms, 2),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if attempt >= settings.cache_model_task_max_retries:
                break
            await asyncio.sleep(min(2 ** (attempt - 1), 8))

    raise HTTPException(
        status_code=500,
        detail={
            "status": "error",
            "hf_model_id": hf_model_id,
            "task_name": task_name,
            "error": str(last_error) if last_error else "unknown cache-model failure",
        },
    )

async def deployment_ready(
    deployment_id: str,
    payload: DeploymentReadyPayload | None = None,
    secret: str | None = None,
    x_visgate_secret: Annotated[
        str | None,
        Header(alias="X-Visgate-Internal-Secret"),
    ] = None,
) -> dict:
    """
    Called by inference container when model is loaded. Sets status=ready and notifies user webhook.
    """
    settings = get_settings()
    provided_secret = x_visgate_secret or secret
    if settings.internal_webhook_secret and provided_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")

    data = payload or DeploymentReadyPayload()

    if data.status == "ready":
        success = await mark_deployment_ready_and_notify(
            deployment_id,
            endpoint_url=data.endpoint_url,
        )
        return {
            "deployment_id": deployment_id,
            "status": "ready",
            "webhook_delivered": success,
        }

    await update_deployment_phase_from_worker(
        deployment_id=deployment_id,
        status=data.status,
        message=data.message,
        endpoint_url=data.endpoint_url,
    )
    return {
        "deployment_id": deployment_id,
        "status": data.status,
        "webhook_delivered": False,
    }


@router.post("/logs/{deployment_id}")
async def log_tunnel(
    deployment_id: str,
    payload: LiveLogPayload,
    secret: str | None = None,
    x_visgate_secret: Annotated[
        str | None,
        Header(alias="X-Visgate-Internal-Secret"),
    ] = None,
) -> dict:
    """Accept live log lines from worker for SSE tunneling."""
    settings = get_settings()
    provided_secret = x_visgate_secret or secret
    if settings.internal_webhook_secret and provided_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")

    append_live_log(deployment_id, payload.level.upper(), payload.message)
    return {"status": "ok"}


@router.post("/cleanup/{deployment_id}")
async def cleanup_endpoint(
    deployment_id: str,
    payload: CleanupPayload | None = None,
    secret: str | None = None,
    x_visgate_secret: Annotated[
        str | None,
        Header(alias="X-Visgate-Internal-Secret"),
    ] = None,
) -> dict:
    """Worker-triggered cleanup to avoid idle billing leaks."""
    settings = get_settings()
    provided_secret = x_visgate_secret or secret
    if settings.internal_webhook_secret and provided_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")

    from src.services.firestore_repo import get_firestore_client
    fs_client = get_firestore_client(settings.gcp_project_id)
    doc = get_deployment(fs_client, settings.firestore_collection_deployments, deployment_id)
    if not doc or not doc.runpod_endpoint_id:
        return {"status": "noop", "reason": "missing_endpoint"}

    # Prefer key sent by the worker (resolves multi-instance secret_cache race).
    # Fall back to in-memory cache for asyncio-based deployments.
    cleanup_body = payload or CleanupPayload()
    runpod_api_key = cleanup_body.runpod_api_key
    if not runpod_api_key:
        cached = get_secrets(deployment_id)
        runpod_api_key = cached.runpod_api_key if cached else None
    if not runpod_api_key:
        return {"status": "noop", "reason": "missing_runpod_key"}

    provider = get_provider(doc.provider or "runpod")
    try:
        await provider.delete_endpoint(doc.runpod_endpoint_id, runpod_api_key)
        update_deployment(fs_client, settings.firestore_collection_deployments, deployment_id, {"status": "deleted"})
        return {"status": "deleted"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@router.post("/model-cached")
async def model_cached(
    payload: ModelCachedPayload,
    secret: str | None = None,
    x_visgate_secret: Annotated[
        str | None,
        Header(alias="X-Visgate-Internal-Secret"),
    ] = None,
) -> dict:
    """
    Called by worker after successfully uploading a model to R2.
    Updates the R2 manifest and marks the deployment as r2_cached.
    """
    settings = get_settings()
    provided_secret = x_visgate_secret or secret
    if settings.internal_webhook_secret and provided_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")

    if not settings.r2_access_key_id_rw or not settings.r2_endpoint_url:
        return {"status": "skipped", "reason": "R2 not configured"}

    try:
        # Update manifest
        ok = add_model_to_manifest(
            model_id=payload.hf_model_id,
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id_rw,
            secret_access_key=settings.r2_secret_access_key_rw,
        )

        # Update deployment doc
        from src.services.firestore_repo import get_firestore_client
        fs_client = get_firestore_client(settings.gcp_project_id)
        update_deployment(
            fs_client,
            settings.firestore_collection_deployments,
            payload.deployment_id,
            {"r2_cached": True},
        )

        return {"status": "ok", "manifest_updated": ok}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
