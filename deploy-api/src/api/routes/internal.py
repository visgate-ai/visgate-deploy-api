"""Internal routes: deployment-ready callback from inference container."""

import asyncio
import json
from typing import Annotated, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.core.config import get_settings
from src.models.schemas import DeploymentReadyPayload
from src.services.deployment import (
    mark_deployment_ready_and_notify,
    orchestrate_deployment,
    update_deployment_phase_from_worker,
)
from src.services.firestore_repo import get_deployment, update_deployment
from src.services.log_tunnel import append_live_log
from src.services.provider_factory import get_provider
from src.services.r2_manifest import add_model_to_manifest
from src.services.secret_cache import get_secrets

router = APIRouter(prefix="/internal", tags=["internal"])


class LiveLogPayload(BaseModel):
    level: str = "INFO"
    message: str


class CleanupPayload(BaseModel):
    reason: str = "idle_timeout"
    runpod_api_key: Optional[str] = None  # Passed by worker; eliminates multi-instance secret_cache dependency


class ModelCachedPayload(BaseModel):
    """Payload for worker → /internal/model-cached callback."""

    hf_model_id: str
    deployment_id: str


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


@router.post("/deployment-ready/{deployment_id}")
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

    if not settings.aws_access_key_id or not settings.aws_endpoint_url:
        return {"status": "skipped", "reason": "R2 not configured"}

    try:
        # Update manifest
        ok = add_model_to_manifest(
            model_id=payload.hf_model_id,
            endpoint_url=settings.aws_endpoint_url,
            access_key_id=settings.aws_access_key_id,
            secret_access_key=settings.aws_secret_access_key,
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
