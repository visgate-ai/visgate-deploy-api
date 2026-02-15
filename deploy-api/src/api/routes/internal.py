"""Internal routes: deployment-ready callback from inference container."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from src.core.config import get_settings
from src.models.schemas import DeploymentReadyPayload
from src.services.deployment import (
    mark_deployment_ready_and_notify,
    update_deployment_phase_from_worker,
)

router = APIRouter(prefix="/internal", tags=["internal"])


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
