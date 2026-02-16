"""Internal routes: deployment-ready callback from inference container."""

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.core.config import get_settings
from src.models.schemas import DeploymentReadyPayload
from src.services.deployment import (
    mark_deployment_ready_and_notify,
    update_deployment_phase_from_worker,
)
from src.services.firestore_repo import get_deployment, update_deployment
from src.services.log_tunnel import append_live_log
from src.services.provider_factory import get_provider
from src.services.secret_cache import get_secrets

router = APIRouter(prefix="/internal", tags=["internal"])


class LiveLogPayload(BaseModel):
    level: str = "INFO"
    message: str


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

    secrets = get_secrets(deployment_id)
    if not secrets:
        return {"status": "noop", "reason": "missing_runpod_key"}

    provider = get_provider(doc.provider or "runpod")
    try:
        await provider.delete_endpoint(doc.runpod_endpoint_id, secrets.runpod_api_key)
        update_deployment(fs_client, settings.firestore_collection_deployments, deployment_id, {"status": "deleted"})
        return {"status": "deleted"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
