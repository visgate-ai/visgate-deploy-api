"""Deployment CRUD: POST /v1/deployments, GET /v1/deployments/{id}, DELETE /v1/deployments/{id}."""

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
    DeploymentCreate,
    DeploymentResponse,
    DeploymentResponse202,
    LogEntrySchema,
)
from src.services.deployment import mark_deployment_ready_and_notify
from src.services.endpoint_naming import model_slug, pool_endpoint_name, user_endpoint_name
from src.services.firestore_repo import get_deployment, set_deployment
from src.services.log_tunnel import get_live_logs_since
from src.services.model_resolver import get_hf_name
from src.services.model_capabilities import supports_task
from src.services.pool_policy import choose_pool_policy
from src.services.provider_factory import get_provider
import src.services.runpod # Register providers
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
        runpod_endpoint_id=doc.runpod_endpoint_id,
        endpoint_url=doc.endpoint_url,
        gpu_allocated=doc.gpu_allocated,
        model_vram_gb=doc.model_vram_gb,
        logs=logs,
        error=doc.error,
        estimated_remaining_seconds=_estimate_remaining_seconds(doc.status),
        phase_durations=_compute_phase_durations(doc),
        created_at=created_at,
        ready_at=ready_at,
    )


def _resolve_runpod_key(body: DeploymentCreate, ctx: RequestContext) -> str:
    if body.user_runpod_key:
        return body.user_runpod_key
    return ctx.runpod_api_key


def _is_warm_status(status: str) -> bool:
    return status.upper() not in {"TERMINATED", "DELETED", "FAILED", "STOPPED"}


def _build_s3_model_url(base_url: str, path_suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{path_suffix.strip('/')}"


def _parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def _resolve_hf_model_id(body: DeploymentCreate) -> str:
    """Resolve to HF model ID: either hf_model_id or get_hf_name(model_name, provider)."""
    if body.hf_model_id:
        if body.model_name:
            raise InvalidDeploymentRequestError(
                "Provide either hf_model_id or model_name (+ optional provider), not both",
            )
        return body.hf_model_id
    if not body.model_name:
        raise InvalidDeploymentRequestError(
            "Provide either hf_model_id or model_name (+ optional provider)",
        )
    return get_hf_name(body.model_name, body.provider)


@router.post("", response_model=DeploymentResponse202, status_code=status.HTTP_202_ACCEPTED)
async def create_deployment(
    body: DeploymentCreate,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> DeploymentResponse202:
    """Create a new deployment (async). Returns 202 with deployment_id; processing continues in background.
    Use either hf_model_id or (model_name + optional provider). deploy_model(hf_name, gpu=auto) -> webhook.
    """
    settings = get_settings()
    from src.core.logging import structured_log
    structured_log(
        "INFO",
        "create_deployment request",
        metadata={
            "model_name": body.model_name,
            "provider": body.provider,
            "gpu_tier": body.gpu_tier,
            "region": body.region,
            "task": body.task,
        },
    )
    hf_model_id = _resolve_hf_model_id(body)
    structured_log("INFO", f"Resolved hf_model_id: {hf_model_id}")
    if body.task and not supports_task(hf_model_id, body.task):
        raise InvalidDeploymentRequestError(
            f"Model {hf_model_id} does not support task {body.task}"
        )
    deployment_id = _generate_deployment_id()
    created_at_dt = datetime.now(UTC)
    created_at = created_at_dt.isoformat().replace("+00:00", "Z")
    stream_url = _build_stream_url(deployment_id)

    runpod_api_key = _resolve_runpod_key(body, ctx)
    if not runpod_api_key:
        raise InvalidDeploymentRequestError("Missing Runpod API key (Authorization header or user_runpod_key)")

    cache_scope = (body.cache_scope or "off").lower()
    if cache_scope not in {"off", "shared", "private"}:
        raise InvalidDeploymentRequestError("cache_scope must be one of: off, shared, private")

    private_fields_present = any(
        [
            body.user_s3_url,
            body.user_aws_access_key_id,
            body.user_aws_secret_access_key,
            body.user_aws_endpoint_url,
        ],
    )
    if cache_scope != "private" and private_fields_present:
        raise InvalidDeploymentRequestError(
            "user_s3_url and user_aws_* fields require cache_scope=private",
        )

    private_s3_url = None
    private_access_key = None
    private_secret_key = None
    private_endpoint = None

    if cache_scope == "shared":
        if not settings.shared_cache_enabled:
            raise InvalidDeploymentRequestError("shared cache is disabled on this service")
        if not settings.s3_model_url:
            raise InvalidDeploymentRequestError("shared cache requires S3_MODEL_URL configured on service")
        allowlisted_models = _parse_csv_set(settings.shared_cache_allowed_models)
        if settings.shared_cache_reject_unlisted and allowlisted_models and hf_model_id not in allowlisted_models:
            raise InvalidDeploymentRequestError(
                "shared cache supports only allowlisted popular models; use cache_scope=off or private",
            )
    if cache_scope == "private":
        if not body.user_s3_url:
            raise InvalidDeploymentRequestError("private cache requires user_s3_url")
        if not body.user_aws_access_key_id or not body.user_aws_secret_access_key:
            raise InvalidDeploymentRequestError("private cache requires user_aws_access_key_id and user_aws_secret_access_key")
        private_s3_url = body.user_s3_url
        private_access_key = body.user_aws_access_key_id
        private_secret_key = body.user_aws_secret_access_key
        private_endpoint = body.user_aws_endpoint_url

    pool_policy = choose_pool_policy(hf_model_id)
    provider = get_provider("runpod")

    # Runpod-native warm discovery
    warm_names = [user_endpoint_name(ctx.user_hash, hf_model_id)]
    if pool_policy.is_warm:
        warm_names.append(pool_endpoint_name(hf_model_id))

    warm_endpoint = None
    try:
        endpoints = await provider.list_endpoints(runpod_api_key)
        for ep in endpoints:
            if ep.get("name") in warm_names and _is_warm_status(ep.get("status", "")):
                warm_endpoint = ep
                break
    except Exception as exc:
        structured_log("WARNING", "Warm discovery failed", metadata={"error": str(exc)})

    if warm_endpoint and warm_endpoint.get("url"):
        reused_doc = DeploymentDoc(
            deployment_id=deployment_id,
            status="ready",
            hf_model_id=hf_model_id,
            user_webhook_url=str(body.user_webhook_url),
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
            webhook_url=str(body.user_webhook_url),
            endpoint_url=warm_endpoint.get("url"),
            path="warm",
            created_at=created_at_dt,
        )

    doc = DeploymentDoc(
        deployment_id=deployment_id,
        status="validating",
        hf_model_id=hf_model_id,
        user_webhook_url=str(body.user_webhook_url),
        gpu_tier=body.gpu_tier,
        created_at=created_at,
        user_hash=ctx.user_hash,
        provider="runpod",
        endpoint_name=user_endpoint_name(ctx.user_hash, hf_model_id),
        pool_policy=pool_policy.name,
        region=body.region,
    )
    set_deployment(firestore_client, settings.firestore_collection_deployments, doc)

    model_path = model_slug(hf_model_id)
    s3_model_url = None
    if cache_scope == "shared" and settings.s3_model_url:
        s3_model_url = _build_s3_model_url(settings.s3_model_url, model_path)
    if cache_scope == "private" and private_s3_url:
        s3_model_url = _build_s3_model_url(private_s3_url, f"{ctx.user_hash}/{model_path}")

    store_secrets(
        deployment_id,
        runpod_api_key,
        body.hf_token,
        aws_access_key_id=private_access_key,
        aws_secret_access_key=private_secret_key,
        aws_endpoint_url=private_endpoint,
        s3_model_url=s3_model_url,
    )
    await enqueue_orchestration_task(
        deployment_id,
        runpod_api_key,
        body.hf_token,
        private_access_key,
        private_secret_key,
        private_endpoint,
        s3_model_url,
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
        webhook_url=str(body.user_webhook_url),
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
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
    ref = firestore_client.collection(settings.firestore_collection_deployments).document(deployment_id)
    ref.update({"status": "deleted"})
