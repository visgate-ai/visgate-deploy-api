"""Pydantic request/response models for API and internal webhooks."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.core.tasks import is_known_task, normalize_task


# --- Request ---
class DeploymentCreate(BaseModel):
    """POST /v1/deployments request body."""

    hf_model_id: str = Field(..., min_length=1, max_length=256, description="Hugging Face model ID, e.g. 'black-forest-labs/FLUX.1-schnell'. See GET /v1/models for supported models.")
    user_runpod_key: str | None = Field(default=None, min_length=1, description="RunPod API key. If omitted, the Authorization Bearer token is used. Providing it here takes precedence.")
    user_webhook_url: HttpUrl | None = Field(default=None, description="Optional URL to notify when deployment is ready")
    gpu_tier: str | None = Field(default=None, max_length=64, description="e.g. A40; auto-select if omitted")
    hf_token: str | None = Field(default=None, description="Hugging Face token owned by the caller. Required for model access and license attribution.")
    region: str | None = Field(default=None, max_length=32, description="Preferred Runpod region/location")
    task: str | None = Field(
        default="text_to_image",
        description="Intended task for compatibility checks",
    )

    @field_validator("task", mode="before")
    @classmethod
    def normalize_task_field(cls, value: str | None) -> str | None:
        normalized = normalize_task(value)
        if normalized and not is_known_task(normalized):
            raise ValueError("Unsupported task")
        return normalized


# --- Response ---
class DeploymentResponse202(BaseModel):
    """202 Accepted response for POST /v1/deployments."""

    deployment_id: str
    status: Literal["warm_ready", "accepted_cold", "validating", "creating_endpoint", "ready"] = "accepted_cold"
    model_id: str
    estimated_ready_seconds: int = Field(default=180, ge=0, le=600)
    estimated_ready_at: datetime
    poll_interval_seconds: int = Field(default=5, ge=1, le=30)
    stream_url: str
    webhook_url: str
    endpoint_url: str | None = None
    path: Literal["warm", "cold"] = "cold"
    created_at: datetime


class LogEntrySchema(BaseModel):
    """Single log entry in deployment response."""

    timestamp: datetime
    level: str
    message: str


DeploymentStatus = Literal[
    "accepted_cold",
    "warm_ready",
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
    hf_model_id: str = ""
    task: str | None = None
    runpod_endpoint_id: str | None = None
    endpoint_url: str | None = None
    gpu_allocated: str | None = None
    model_vram_gb: int | None = None
    logs: list[LogEntrySchema] = Field(default_factory=list)
    error: str | None = None
    estimated_remaining_seconds: int | None = None
    phase_durations: dict[str, float] = Field(default_factory=dict)
    t_r2_sync_s: float | None = None
    t_model_load_s: float | None = None
    loaded_from_cache: bool | None = None
    created_at: datetime
    ready_at: datetime | None = None


class DeploymentListResponse(BaseModel):
    """GET /v1/deployments response."""

    deployments: list[DeploymentResponse]
    total: int
    limit: int


class GpuTypeInfo(BaseModel):
    """Information for a specific GPU type on RunPod."""
    id: str
    display_name: str
    memory_gb: int
    secure_cloud: bool
    community_cloud: bool
    bid_price_per_hr: float | None = None
    price_per_hr: float | None = None


class GpuListResponse(BaseModel):
    """GET /v1/deployments/gpus response."""
    gpus: list[GpuTypeInfo]


class ModelEntry(BaseModel):
    """Single entry in GET /v1/models response."""

    model_id: str
    tasks: list[str]
    gpu_memory_gb: int
    cached: bool = False  # True when model weights are stored in platform R2 cache


class ModelsListResponse(BaseModel):
    """GET /v1/models response."""

    models: list[ModelEntry]
    total: int
    cache_enabled: bool


class HFModelResult(BaseModel):
    """Single entry in HuggingFace model search."""
    model_id: str
    task: str | None = None
    downloads: int = 0
    likes: int = 0
    vram_required_gb: int | None = None


class HFModelSearchResponse(BaseModel):
    """Response from GET /v1/models/search."""
    results: list[HFModelResult]
    query: str


class ValidateKeyRequest(BaseModel):
    """Request body for provider key validation."""

    provider: str = Field(..., min_length=1, max_length=64)
    api_key: str = Field(..., min_length=1, max_length=4096)


class ValidateKeyResponse(BaseModel):
    """Validation result for a provider API key."""

    valid: bool
    message: str


class DeploymentCostResponse(BaseModel):
    """Estimate GPU deployment cost."""
    deployment_id: str
    status: str
    gpu_allocated: str | None = None
    hours_running: float | None = None
    price_per_hour_usd: float | None = None
    estimated_cost_usd: float | None = None
    note: str | None = None


# --- Internal webhook (container -> orchestrator) ---
class DeploymentReadyPayload(BaseModel):
    """POST /internal/deployment-ready/{deployment_id} optional body."""

    status: Literal["downloading_model", "loading_model", "ready", "failed"] = "ready"
    message: str | None = None
    endpoint_url: str | None = None
    t_r2_sync_s: float | None = None
    t_model_load_s: float | None = None
    loaded_from_cache: bool | None = None


InferenceJobStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "expired",
]


class InferencePolicy(BaseModel):
    execution_timeout_ms: int | None = Field(default=None, ge=5000, le=604800000)
    ttl_ms: int | None = Field(default=None, ge=10000, le=604800000)
    low_priority: bool = False


class InferenceOutputDestination(BaseModel):
    bucket_name: str
    endpoint_url: str
    key_prefix: str | None = None


class InferenceArtifactMetadata(BaseModel):
    bucket_name: str | None = None
    endpoint_url: str | None = None
    key: str | None = None
    url: str | None = None
    content_type: str | None = None
    bytes: int | None = None


class InferenceJobMetrics(BaseModel):
    queue_ms: int | None = None
    execution_ms: int | None = None
    wall_clock_ms: int | None = None


class InferenceJobCreate(BaseModel):
    deployment_id: str = Field(..., min_length=1)
    task: str | None = Field(default=None, description="Optional task override; normalized to canonical task names")
    input: dict[str, Any] = Field(default_factory=dict)
    user_webhook_url: HttpUrl | None = Field(default=None, description="Optional user webhook for job completion notifications")
    policy: InferencePolicy | None = None

    @field_validator("task", mode="before")
    @classmethod
    def normalize_job_task(cls, value: str | None) -> str | None:
        normalized = normalize_task(value)
        if normalized and not is_known_task(normalized):
            raise ValueError("Unsupported task")
        return normalized


class InferenceJobResponse(BaseModel):
    job_id: str
    deployment_id: str
    provider: str
    provider_job_id: str | None = None
    task: str | None = None
    status: InferenceJobStatus
    provider_status: str | None = None
    endpoint_url: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output_destination: InferenceOutputDestination | None = None
    artifact: InferenceArtifactMetadata | None = None
    metrics: InferenceJobMetrics | None = None
    estimated_cost_usd: float | None = None
    output: Any = None
    output_preview: Any = None
    error: Any = None
    progress: Any = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    user_webhook_url: HttpUrl | None = None


class InferenceJobAcceptedResponse(BaseModel):
    job_id: str
    deployment_id: str
    provider: str
    provider_job_id: str
    status: InferenceJobStatus = "queued"
    provider_status: str
    output_destination: InferenceOutputDestination | None = None


class InferenceJobListResponse(BaseModel):
    jobs: list[InferenceJobResponse]
    total: int


class InferenceJobWebhookPayload(BaseModel):
    id: str | None = None
    status: str | None = None
    output: Any = None
    error: Any = None
    delayTime: int | None = None
    executionTime: int | None = None

