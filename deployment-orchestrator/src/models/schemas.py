"""Pydantic request/response models for API and internal webhooks."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


# --- Request ---
class DeploymentCreate(BaseModel):
    """POST /v1/deployments request body.

    Either hf_model_id OR (model_name + optional provider) must be set.
    Flow: model_name = get_models(provider=fal, "veo3") [external] -> hf_name = get_hf_name(model_name, provider) -> deploy_model(hf_name, gpu=auto) -> webhook.
    """

    hf_model_id: Optional[str] = Field(default=None, min_length=1, max_length=256, description="Hugging Face model ID (or use model_name+provider)")
    model_name: Optional[str] = Field(default=None, min_length=1, max_length=128, description="Model name from get_models (e.g. veo3); resolved to HF ID via get_hf_name")
    provider: Optional[str] = Field(default=None, max_length=64, description="Provider from get_models (e.g. fal)")
    user_runpod_key: str = Field(..., min_length=1, description="Runpod API key or secret ref")
    user_webhook_url: HttpUrl = Field(..., description="URL to notify when deployment is ready")
    gpu_tier: Optional[str] = Field(default=None, max_length=64, description="e.g. A40; auto-select if omitted")
    hf_token: Optional[str] = Field(default=None, description="Optional HF token for gated models")


# --- Response ---
class DeploymentResponse202(BaseModel):
    """202 Accepted response for POST /v1/deployments."""

    deployment_id: str
    status: Literal["validating"] = "validating"
    model_id: str
    estimated_ready_seconds: int = Field(default=180, ge=60, le=600)
    webhook_url: str
    created_at: datetime


class LogEntrySchema(BaseModel):
    """Single log entry in deployment response."""

    timestamp: datetime
    level: str
    message: str


DeploymentStatus = Literal[
    "validating",
    "selecting_gpu",
    "creating_endpoint",
    "downloading_model",
    "loading_model",
    "ready",
    "failed",
    "webhook_failed",
    "deleted",
]


class DeploymentResponse(BaseModel):
    """GET /v1/deployments/{id} response."""

    deployment_id: str
    status: DeploymentStatus
    runpod_endpoint_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    gpu_allocated: Optional[str] = None
    model_vram_gb: Optional[int] = None
    logs: list[LogEntrySchema] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime
    ready_at: Optional[datetime] = None


# --- Internal webhook (container -> orchestrator) ---
class DeploymentReadyPayload(BaseModel):
    """POST /internal/deployment-ready/{deployment_id} optional body."""

    status: Literal["ready"] = "ready"
    message: Optional[str] = None
    endpoint_url: Optional[str] = None
