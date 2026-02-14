"""Deployment CRUD: POST /v1/deployments, GET /v1/deployments/{id}, DELETE /v1/deployments/{id}."""

import asyncio
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, status

from src.api.dependencies import get_current_api_key, get_firestore
from src.core.config import get_settings
from src.core.errors import DeploymentNotFoundError, InvalidDeploymentRequestError
from src.core.telemetry import record_deployment_created
from src.models.entities import DeploymentDoc
from src.models.schemas import (
    DeploymentCreate,
    DeploymentResponse,
    DeploymentResponse202,
    LogEntrySchema,
)
from src.services.deployment import orchestrate_deployment
from src.services.firestore_repo import get_deployment, set_deployment
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
    hf_model_id = _resolve_hf_model_id(body)
    deployment_id = _generate_deployment_id()
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

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
        webhook_url=str(body.user_webhook_url),
        created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
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
