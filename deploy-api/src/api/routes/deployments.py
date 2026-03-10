"""Deployment CRUD: GET/POST /v1/deployments, GET /v1/deployments/{id}, DELETE /v1/deployments/{id}."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import RequestContext, get_firestore, get_request_context
from src.core.config import get_settings
from src.core.errors import DeploymentNotFoundError, InvalidDeploymentRequestError
from src.core.telemetry import record_deployment_created
from src.models.entities import DeploymentDoc, LogEntry
from src.models.schemas import (
    DeploymentCostResponse,
    DeploymentCreate,
    DeploymentListResponse,
    DeploymentResponse,
    DeploymentResponse202,
    GpuListResponse,
    GpuTypeInfo,
    LogEntrySchema,
)
from src.services.deployment import mark_deployment_ready_and_notify
from src.services.endpoint_naming import model_slug, pool_endpoint_name, user_endpoint_name
from src.services.internal_urls import resolve_internal_base_url
from src.services.log_tunnel import get_live_logs_since

def _get_repo():
    if get_settings().effective_use_memory_repo:
        import src.services.memory_repo as repo
    else:
        import src.services.firestore_repo as repo
    return repo

def get_deployment(*args, **kwargs):
    return _get_repo().get_deployment(*args, **kwargs)

def list_deployments(*args, **kwargs):
    return _get_repo().list_deployments(*args, **kwargs)

def set_deployment(*args, **kwargs):
    return _get_repo().set_deployment(*args, **kwargs)
from src.services.model_capabilities import supports_task
from src.services.pool_policy import choose_pool_policy
from src.services.provider_factory import get_provider
from src.services.secret_cache import store_secrets
from src.services.tasks import enqueue_orchestration_task

router = APIRouter(prefix="/v1/deployments", tags=["deployments"])


def _generate_deployment_id() -> str:
    """Generate unique deployment ID (e.g. dep_2024_abc123)."""
    import secrets
    from datetime import datetime
    y = datetime.now(UTC).strftime("%Y")
    suffix = secrets.token_hex(4)
    return f"dep_{y}_{suffix}"


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _estimate_remaining_seconds(status: str) -> int:
    mapping = {
        "validating": 20,
        "selecting_gpu": 15,
        "creating_endpoint": 120,
        "downloading_model": 90,
        "loading_model": 45,
        "ready": 0,
        "failed": 0,
        "webhook_failed": 0,
        "deleted": 0,
    }
    return mapping.get(status, 60)


def _compute_phase_durations(doc: DeploymentDoc) -> dict[str, float]:
    created_at = _parse_iso(doc.created_at)
    ready_at = _parse_iso(doc.ready_at)
    if not created_at:
        return {}
    end_time = ready_at or datetime.now(UTC)
    total_seconds = max((end_time - created_at).total_seconds(), 0.0)
    return {"total_elapsed_seconds": round(total_seconds, 2)}


def _build_stream_url(deployment_id: str) -> str:
    return f"/v1/deployments/{deployment_id}/stream"


def _doc_to_response(doc: DeploymentDoc) -> DeploymentResponse:
    """Convert Firestore doc to GET response schema."""
    created_at = _parse_iso(doc.created_at) or datetime.now(UTC)
    ready_at = _parse_iso(doc.ready_at)
    logs = [
        LogEntrySchema(
            timestamp=_parse_iso(e.timestamp) or created_at,
            level=e.level,
            message=e.message,
        )
        for e in doc.logs
    ]
    return DeploymentResponse(
        deployment_id=doc.deployment_id,
        status=doc.status,
        hf_model_id=doc.hf_model_id,
        task=doc.task,
        runpod_endpoint_id=doc.runpod_endpoint_id,
        endpoint_url=doc.endpoint_url,
        gpu_allocated=doc.gpu_allocated,
        model_vram_gb=doc.model_vram_gb,
        logs=logs,
        error=doc.error,
        estimated_remaining_seconds=_estimate_remaining_seconds(doc.status),
        phase_durations=_compute_phase_durations(doc),
        t_r2_sync_s=doc.t_r2_sync_s,
        t_model_load_s=doc.t_model_load_s,
        loaded_from_cache=doc.loaded_from_cache,
        created_at=created_at,
        ready_at=ready_at,
    )


def _resolve_runpod_key(body: DeploymentCreate, ctx: RequestContext) -> str:
    if body.user_runpod_key:
        return body.user_runpod_key
    return ctx.runpod_api_key


def _is_warm_status(status: str) -> bool:
    return status.upper() not in {"TERMINATED", "DELETED", "FAILED", "STOPPED"}


def _has_warm_worker(health: dict) -> bool:
    workers = health.get("workers") or {}
    ready = int(workers.get("ready", 0) or 0)
    idle = int(workers.get("idle", 0) or 0)
    return ready > 0 or idle > 0


def _build_s3_model_url(base_url: str, path_suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{path_suffix.strip('/')}"


def _parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


@router.get("/gpus", response_model=GpuListResponse, summary="List available GPU types")
async def list_gpus(
    ctx: Annotated[RequestContext, Depends(get_request_context)],
) -> GpuListResponse:
    """List available GPU types with VRAM and current pricing from RunPod."""
    provider = get_provider("runpod")
    raw_gpus = await provider.list_gpu_types(ctx.runpod_api_key)

    gpus: list[GpuTypeInfo] = []
    for g in raw_gpus:
        gpus.append(
            GpuTypeInfo(
                id=g.get("id", ""),
                display_name=g.get("displayName", ""),
                memory_gb=g.get("memoryInGb") or 0,
                secure_cloud=g.get("secureCloud") or False,
                community_cloud=g.get("communityCloud") or False,
                bid_price_per_hr=g.get("communityPrice"),
                price_per_hr=g.get("securePrice"),
            )
        )
    # Sort by memory, then price
    gpus.sort(key=lambda x: (x.memory_gb, x.price_per_hr or 999))
    return GpuListResponse(gpus=gpus)


@router.get("", response_model=DeploymentListResponse, summary="List deployments")
async def list_deployments_route(
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
    deployment_status: str | None = None,
    limit: int = 20,
) -> DeploymentListResponse:
    """List deployments for the authenticated API key, newest first.

    Use ``deployment_status`` to filter by status (e.g. ``ready``, ``failed``).
    ``limit`` is capped at 100.
    """
    settings = get_settings()
    limit = max(1, min(limit, 100))
    docs = list_deployments(
        firestore_client,
        settings.firestore_collection_deployments,
        user_hash=ctx.user_hash,
        status_filter=deployment_status,
        limit=limit,
    )
    items = [_doc_to_response(doc) for doc in docs]
    return DeploymentListResponse(deployments=items, total=len(items), limit=limit)


@router.post("", response_model=DeploymentResponse202, status_code=status.HTTP_202_ACCEPTED)
async def create_deployment(
    request: Request,
    body: DeploymentCreate,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> DeploymentResponse202:
    """Create a new deployment (async). Returns 202 with deployment_id; processing continues in background."""
    settings = get_settings()
    from src.core.logging import structured_log
    structured_log(
        "INFO",
        "create_deployment request",
        metadata={
            "hf_model_id": body.hf_model_id,
            "gpu_tier": body.gpu_tier,
            "region": body.region,
            "task": body.task,
        },
    )
    hf_model_id = body.hf_model_id
    structured_log("INFO", f"hf_model_id: {hf_model_id}")
    if body.task and not supports_task(hf_model_id, body.task):
        raise InvalidDeploymentRequestError(
            f"Model {hf_model_id} does not support task {body.task}"
        )
    deployment_id = _generate_deployment_id()
    created_at_dt = datetime.now(UTC)
    created_at = created_at_dt.isoformat().replace("+00:00", "Z")
    stream_url = _build_stream_url(deployment_id)
    internal_webhook_base_url = resolve_internal_base_url(request)

    runpod_api_key = _resolve_runpod_key(body, ctx)
    if not runpod_api_key:
        raise InvalidDeploymentRequestError("Missing Runpod API key (Authorization header or user_runpod_key)")
    if not body.hf_token:
        raise InvalidDeploymentRequestError("hf_token is required and must belong to the caller")

    pool_policy = choose_pool_policy(hf_model_id)
    provider = get_provider("runpod")

    # Runpod-native warm discovery
    warm_names = [user_endpoint_name(ctx.user_hash, hf_model_id)]
    if pool_policy.is_warm:
        warm_names.append(pool_endpoint_name(hf_model_id))

    warm_endpoint = None
    if settings.enable_endpoint_reuse:
        try:
            endpoints = await provider.list_endpoints(runpod_api_key)
            for ep in endpoints:
                if ep.get("name") not in warm_names:
                    continue
                if not _is_warm_status(ep.get("status", "")):
                    continue

                endpoint_id = ep.get("id")
                if not endpoint_id:
                    continue

                health = await provider.check_endpoint_health(endpoint_id, runpod_api_key)
                if _has_warm_worker(health):
                    warm_endpoint = ep
                    break

                structured_log(
                    "INFO",
                    "Warm endpoint skipped due to no ready/idle workers",
                    metadata={"endpoint_id": endpoint_id, "name": ep.get("name")},
                )
        except Exception as exc:
            structured_log("WARNING", "Warm discovery failed", metadata={"error": str(exc)})

    if warm_endpoint and warm_endpoint.get("url"):
        reused_doc = DeploymentDoc(
            deployment_id=deployment_id,
            status="ready",
            hf_model_id=hf_model_id,
            user_webhook_url=str(body.user_webhook_url) if body.user_webhook_url else None,
            gpu_tier=body.gpu_tier,
            runpod_endpoint_id=warm_endpoint.get("id"),
            endpoint_url=warm_endpoint.get("url"),
            gpu_allocated=None,
            model_vram_gb=None,
            logs=[
                LogEntry(
                    timestamp=created_at,
                    level="INFO",
                    message=f"Warm endpoint matched: {warm_endpoint.get('id')}",
                )
            ],
            created_at=created_at,
            user_hash=ctx.user_hash,
            provider="runpod",
            endpoint_name=warm_endpoint.get("name"),
            pool_policy=pool_policy.name,
            region=body.region,
            task=body.task,
            internal_webhook_base_url=internal_webhook_base_url,
        )
        set_deployment(firestore_client, settings.firestore_collection_deployments, reused_doc)
        await mark_deployment_ready_and_notify(
            deployment_id,
            endpoint_url=warm_endpoint.get("url"),
        )
        record_deployment_created()
        return DeploymentResponse202(
            deployment_id=deployment_id,
            status="warm_ready",
            model_id=hf_model_id,
            estimated_ready_seconds=0,
            estimated_ready_at=created_at_dt,
            poll_interval_seconds=1,
            stream_url=stream_url,
            webhook_url=str(body.user_webhook_url) if body.user_webhook_url else "",
            endpoint_url=warm_endpoint.get("url"),
            path="warm",
            created_at=created_at_dt,
        )

    doc = DeploymentDoc(
        deployment_id=deployment_id,
        status="validating",
        hf_model_id=hf_model_id,
        user_webhook_url=str(body.user_webhook_url) if body.user_webhook_url else None,
        gpu_tier=body.gpu_tier,
        created_at=created_at,
        user_hash=ctx.user_hash,
        provider="runpod",
        endpoint_name=user_endpoint_name(ctx.user_hash, hf_model_id),
        pool_policy=pool_policy.name,
        region=body.region,
        task=body.task,
        internal_webhook_base_url=internal_webhook_base_url,
    )
    set_deployment(firestore_client, settings.firestore_collection_deployments, doc)

    store_secrets(
        deployment_id,
        runpod_api_key,
        body.hf_token,
    )
    await enqueue_orchestration_task(
        deployment_id,
        runpod_api_key,
        body.hf_token,
    )
    record_deployment_created()

    return DeploymentResponse202(
        deployment_id=deployment_id,
        status="accepted_cold",
        model_id=hf_model_id,
        estimated_ready_seconds=180,
        estimated_ready_at=created_at_dt + timedelta(seconds=180),
        poll_interval_seconds=5,
        stream_url=stream_url,
        webhook_url=str(body.user_webhook_url) if body.user_webhook_url else "",
        path="cold",
        created_at=created_at_dt,
    )


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment_status(
    deployment_id: str,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> DeploymentResponse:
    """Get deployment status and logs."""
    settings = get_settings()
    doc = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
    if not doc:
        raise DeploymentNotFoundError(deployment_id)
    if doc.user_hash and doc.user_hash != ctx.user_hash:
        raise DeploymentNotFoundError(deployment_id)
    return _doc_to_response(doc)


@router.get("/{deployment_id}/stream")
async def stream_deployment_status(
    deployment_id: str,
    request: Request,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
):
    """Stream deployment status updates as SSE."""
    settings = get_settings()

    async def event_generator():
        last_status = None
        while True:
            if await request.is_disconnected():
                break

            doc = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
            if not doc or (doc.user_hash and doc.user_hash != ctx.user_hash):
                payload = {"deployment_id": deployment_id, "error": "not_found"}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                break

            if doc.status != last_status:
                payload = {
                    "deployment_id": deployment_id,
                    "status": doc.status,
                    "endpoint_url": doc.endpoint_url,
                    "error": doc.error,
                    "estimated_remaining_seconds": _estimate_remaining_seconds(doc.status),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }
                yield f"event: status\ndata: {json.dumps(payload)}\n\n"
                last_status = doc.status

            if doc.status in {"ready", "failed", "webhook_failed", "deleted"}:
                break
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{deployment_id}/logs/stream")
async def stream_deployment_logs(
    deployment_id: str,
    request: Request,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
):
    """Stream live log lines from worker (stateless tunnel)."""
    settings = get_settings()
    doc = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
    if not doc or (doc.user_hash and doc.user_hash != ctx.user_hash):
        raise DeploymentNotFoundError(deployment_id)

    async def event_generator():
        last_ts = 0.0
        while True:
            if await request.is_disconnected():
                break
            entries = get_live_logs_since(deployment_id, last_ts)
            for entry in entries:
                payload = {
                    "deployment_id": deployment_id,
                    "timestamp": entry.timestamp,
                    "level": entry.level,
                    "message": entry.message,
                }
                last_ts = max(last_ts, entry.timestamp)
                yield f"event: log\ndata: {json.dumps(payload)}\n\n"
            # Stop streaming once deployment reaches a terminal state
            current = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
            if current and current.status in {"ready", "failed", "webhook_failed", "deleted"}:
                yield f"event: done\ndata: {json.dumps({'deployment_id': deployment_id, 'status': current.status})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{deployment_id}/cost", response_model=DeploymentCostResponse, summary="Estimate deployment cost")
async def get_deployment_cost(
    deployment_id: str,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> DeploymentCostResponse:
    """Estimate GPU deployment cost using Firestore record and live RunPod pricing."""
    settings = get_settings()
    doc = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
    if not doc:
        raise DeploymentNotFoundError(deployment_id)
    if doc.user_hash and doc.user_hash != ctx.user_hash:
        raise DeploymentNotFoundError(deployment_id)

    status_val = doc.status
    gpu_allocated = doc.gpu_allocated
    ready_at = _parse_iso(doc.ready_at)
    created_at = _parse_iso(doc.created_at)

    hours_running: float | None = None
    if ready_at:
        end_dt = datetime.now(UTC) if status_val not in ("deleted", "failed") else ready_at
        hours_running = max((end_dt - ready_at).total_seconds() / 3600, 0.0)
    elif created_at and status_val == "ready":
        # Fallback to created_at if ready_at missing but status is ready
        hours_running = max((datetime.now(UTC) - created_at).total_seconds() / 3600, 0.0)

    # Fetch GPU pricing from RunPod
    price_per_hour: float | None = None
    note: str | None = None

    if gpu_allocated:
        provider = get_provider("runpod")
        try:
            gpu_types = await provider.list_gpu_types(ctx.runpod_api_key)
            for g in gpu_types:
                name = g.get("displayName", "").lower()
                gpu_id = g.get("id", "").lower()
                alloc = gpu_allocated.lower()
                if alloc in name or name in alloc or alloc == gpu_id:
                     price_per_hour = g.get("communityPrice") or g.get("securePrice")
                     break
            if price_per_hour is None:
                note = f"GPU type '{gpu_allocated}' not found in current pricing list"
        except Exception as e:
            note = f"Could not fetch GPU pricing: {str(e)[:100]}"
    else:
        note = "GPU not yet allocated (deployment may be starting or failed)"

    estimated_cost: float | None = None
    if hours_running is not None and price_per_hour is not None:
        estimated_cost = round(hours_running * price_per_hour, 6)

    return DeploymentCostResponse(
        deployment_id=deployment_id,
        status=status_val,
        gpu_allocated=gpu_allocated,
        hours_running=round(hours_running, 4) if hours_running is not None else None,
        price_per_hour_usd=price_per_hour,
        estimated_cost_usd=estimated_cost,
        note=note,
    )


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deployment(
    deployment_id: str,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> None:
    """Delete deployment and tear down Runpod endpoint."""
    settings = get_settings()
    doc = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
    if not doc:
        raise DeploymentNotFoundError(deployment_id)
    if doc.user_hash and doc.user_hash != ctx.user_hash:
        raise DeploymentNotFoundError(deployment_id)
    if doc.runpod_endpoint_id:
        try:
            # For now default to runpod, but doc could store the provider name
            provider = get_provider("runpod")
            await provider.delete_endpoint(doc.runpod_endpoint_id, ctx.runpod_api_key)
        except Exception:  # nosec B110 — best-effort cleanup, deletion failure must not block response
            pass
    if doc.runpod_dep_template_name:
        try:
            provider = get_provider("runpod")
            await provider.delete_template(doc.runpod_dep_template_name, ctx.runpod_api_key)
        except Exception:  # nosec B110 — best-effort; template may still be in use briefly after endpoint deletion
            pass
    ref = firestore_client.collection(settings.firestore_collection_deployments).document(deployment_id)
    ref.update({"status": "deleted"})
