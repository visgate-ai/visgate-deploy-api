"""Deployment CRUD: POST /v1/deployments, GET /v1/deployments/{id}, DELETE /v1/deployments/{id}."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_current_api_key, get_firestore
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
from src.services.firestore_repo import (
    find_reusable_deployment,
    get_deployment,
    set_deployment,
)
from src.services.model_resolver import get_hf_name
from src.services.provider_factory import get_provider
import src.services.runpod # Register providers
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
    api_key_id: Annotated[str, Depends(get_current_api_key)],
    firestore_client=Depends(get_firestore),
) -> DeploymentResponse202:
    """Create a new deployment (async). Returns 202 with deployment_id; processing continues in background.
    Use either hf_model_id or (model_name + optional provider). deploy_model(hf_name, gpu=auto) -> webhook.
    """
    settings = get_settings()
    from src.core.logging import structured_log
    structured_log("INFO", f"create_deployment body: {body.model_dump()}")
    hf_model_id = _resolve_hf_model_id(body)
    structured_log("INFO", f"Resolved hf_model_id: {hf_model_id}")
    deployment_id = _generate_deployment_id()
    created_at_dt = datetime.now(UTC)
    created_at = created_at_dt.isoformat().replace("+00:00", "Z")
    stream_url = _build_stream_url(deployment_id)

    reusable = None
    if settings.enable_endpoint_reuse:
        reusable = find_reusable_deployment(
            firestore_client,
            settings.firestore_collection_deployments,
            api_key_id=api_key_id,
            hf_model_id=hf_model_id,
            gpu_tier=body.gpu_tier,
            user_runpod_key=body.user_runpod_key,
        )
    if reusable:
        reused_doc = DeploymentDoc(
            deployment_id=deployment_id,
            status="creating_endpoint",
            hf_model_id=hf_model_id,
            user_runpod_key_ref=body.user_runpod_key,
            user_webhook_url=str(body.user_webhook_url),
            gpu_tier=body.gpu_tier,
            hf_token_ref=body.hf_token,
            runpod_endpoint_id=reusable.runpod_endpoint_id,
            endpoint_url=reusable.endpoint_url,
            gpu_allocated=reusable.gpu_allocated,
            model_vram_gb=reusable.model_vram_gb,
            logs=[
                LogEntry(
                    timestamp=created_at,
                    level="INFO",
                    message=f"Reusing active endpoint {reusable.runpod_endpoint_id}",
                )
            ],
            created_at=created_at,
            api_key_id=api_key_id,
        )
        set_deployment(firestore_client, settings.firestore_collection_deployments, reused_doc)
        await mark_deployment_ready_and_notify(
            deployment_id,
            endpoint_url=reusable.endpoint_url,
        )
        record_deployment_created()
        return DeploymentResponse202(
            deployment_id=deployment_id,
            status="ready",
            model_id=hf_model_id,
            estimated_ready_seconds=0,
            estimated_ready_at=created_at_dt,
            poll_interval_seconds=1,
            stream_url=stream_url,
            webhook_url=str(body.user_webhook_url),
            endpoint_url=reusable.endpoint_url,
            created_at=created_at_dt,
        )

    doc = DeploymentDoc(
        deployment_id=deployment_id,
        status="validating",
        hf_model_id=hf_model_id,
        user_runpod_key_ref=body.user_runpod_key,
        user_webhook_url=str(body.user_webhook_url),
        gpu_tier=body.gpu_tier,
        hf_token_ref=body.hf_token,
        created_at=created_at,
        api_key_id=api_key_id,
    )
    set_deployment(firestore_client, settings.firestore_collection_deployments, doc)

    await enqueue_orchestration_task(deployment_id)
    record_deployment_created()

    return DeploymentResponse202(
        deployment_id=deployment_id,
        status="validating",
        model_id=hf_model_id,
        estimated_ready_seconds=180,
        estimated_ready_at=created_at_dt + timedelta(seconds=180),
        poll_interval_seconds=5,
        stream_url=stream_url,
        webhook_url=str(body.user_webhook_url),
        created_at=created_at_dt,
    )


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment_status(
    deployment_id: str,
    api_key_id: Annotated[str, Depends(get_current_api_key)],
    firestore_client=Depends(get_firestore),
) -> DeploymentResponse:
    """Get deployment status and logs."""
    settings = get_settings()
    doc = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
    if not doc:
        raise DeploymentNotFoundError(deployment_id)
    if doc.api_key_id and doc.api_key_id != api_key_id:
        raise DeploymentNotFoundError(deployment_id)
    return _doc_to_response(doc)


@router.get("/{deployment_id}/stream")
async def stream_deployment_status(
    deployment_id: str,
    request: Request,
    api_key_id: Annotated[str, Depends(get_current_api_key)],
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
            if not doc or (doc.api_key_id and doc.api_key_id != api_key_id):
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


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deployment(
    deployment_id: str,
    api_key_id: Annotated[str, Depends(get_current_api_key)],
    firestore_client=Depends(get_firestore),
) -> None:
    """Delete deployment and tear down Runpod endpoint."""
    settings = get_settings()
    doc = get_deployment(firestore_client, settings.firestore_collection_deployments, deployment_id)
    if not doc:
        raise DeploymentNotFoundError(deployment_id)
    if doc.api_key_id and doc.api_key_id != api_key_id:
        raise DeploymentNotFoundError(deployment_id)
    if doc.runpod_endpoint_id and doc.user_runpod_key_ref:
        try:
            # For now default to runpod, but doc could store the provider name
            provider = get_provider("runpod")
            await provider.delete_endpoint(doc.runpod_endpoint_id, doc.user_runpod_key_ref)
        except Exception:
            pass
    ref = firestore_client.collection(settings.firestore_collection_deployments).document(deployment_id)
    ref.update({"status": "deleted"})
