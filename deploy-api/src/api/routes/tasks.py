"""Internal tasks triggered by Cloud Tasks."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.api.dependencies import verify_internal_webhook_secret
from src.core.logging import structured_log
from src.services.deployment import orchestrate_deployment

router = APIRouter(prefix="/internal/tasks", tags=["tasks"])


class OrchestrateDeploymentRequest(BaseModel):
    deployment_id: str
    runpod_api_key: str
    hf_token: str | None = None


@router.post(
    "/orchestrate-deployment",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_internal_webhook_secret)],
)
async def task_orchestrate_deployment(body: OrchestrateDeploymentRequest) -> dict[str, str]:
    """
    Execute orchestration logic.
    Called by Cloud Tasks. Must return 2xx to acknowledge success.
    If it fails/raises, Cloud Tasks will retry based on queue config.
    """
    structured_log(
        "INFO",
        "Received orchestration task",
        deployment_id=body.deployment_id,
        operation="task.orchestrate",
    )
    
    # Run synchronously so Cloud Run keeps instance alive
    await orchestrate_deployment(body.deployment_id, body.runpod_api_key, body.hf_token)
    
    return {"status": "ok"}
