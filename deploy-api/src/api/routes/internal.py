"""Internal routes: deployment-ready callback from inference container."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from src.core.config import get_settings
from src.services.deployment import mark_deployment_ready_and_notify

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post("/deployment-ready/{deployment_id}")
async def deployment_ready(
    deployment_id: str,
    x_visgate_secret: Annotated[
        str | None,
        Header(alias="X-Visgate-Internal-Secret"),
    ] = None,
) -> dict:
    """
    Called by inference container when model is loaded. Sets status=ready and notifies user webhook.
    """
    settings = get_settings()
    if settings.internal_webhook_secret and x_visgate_secret != settings.internal_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid internal secret")

    success = await mark_deployment_ready_and_notify(deployment_id)
    return {"deployment_id": deployment_id, "status": "ready", "webhook_delivered": success}
