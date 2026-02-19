"""Application settings loaded from environment with validation."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Orchestrator settings from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # GCP
    gcp_project_id: str = Field(..., min_length=1, description="GCP project ID")
    gcp_location: str = Field(default="europe-west1", description="GCP region for Cloud Tasks")
    cloud_tasks_queue_path: str = Field(
        default="",
        description="Full path to Cloud Tasks queue (projects/.../locations/.../queues/...)",
    )

    # Firestore
    firestore_collection_deployments: str = Field(
        default="deployments",
        min_length=1,
        description="Firestore collection for deployment documents",
    )
    firestore_collection_logs: str = Field(
        default="deployment_logs",
        min_length=1,
        description="Firestore collection for deployment audit logs",
    )
    firestore_collection_api_keys: str = Field(
        default="api_keys",
        min_length=1,
        description="Firestore collection for API keys",
    )
    firestore_collection_gpu_registry: str = Field(
        default="gpu_registry",
        min_length=1,
        description="Firestore collection for GPU specifications",
    )
    firestore_collection_gpu_tiers: str = Field(
        default="gpu_tiers",
        min_length=1,
        description="Firestore collection for GPU tier mappings",
    )

    # Runpod
    docker_image: str = Field(
        default="uzunenes/inference:latest",
        min_length=1,
        description="Docker image for Runpod inference workers",
    )
    runpod_template_id: str = Field(
        default="",
        description="Runpod template ID for the inference image (create in Runpod console)",
    )
    runpod_volume_size_gb: int = Field(
        default=0,
        description="Optional persistent volume size in GB (0 to disable)",
    )
    runpod_max_retries: int = Field(default=3, ge=1, le=10)
    runpod_graphql_url: str = Field(
        default="https://api.runpod.io/graphql",
        description="Runpod GraphQL API URL",
    )
    runpod_default_locations: str = Field(
        default="US",
        description="Default Runpod locations (comma-separated)",
    )
    runpod_workers_min: int = Field(
        default=0,
        ge=0,
        le=10,
        description=(
            "Minimum warm workers per endpoint. "
            "0 = no idle cost (cold start on each job, ~40-60s for large models). "
            "1 = one always-warm worker (~$0.35-0.80/hr idle cost per deployment). "
            "Use 1 only when sub-second response latency is required."
        ),
    )
    runpod_workers_max: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum concurrent workers per endpoint",
    )
    runpod_idle_timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=3600,
        description=(
            "Seconds a worker stays alive after its last job. "
            "120s is a good balance: absorbs burst traffic without burning idle GPU hours. "
            "Lower = cheaper but more cold starts; higher = faster burst but more idle cost."
        ),
    )
    runpod_scaler_type: str = Field(
        default="QUEUE_DELAY",
        description="RunPod autoscaler type: QUEUE_DELAY (scale on queue wait) or REQUEST_COUNT",
    )
    runpod_scaler_value: int = Field(
        default=1,
        ge=1,
        le=60,
        description=(
            "QUEUE_DELAY seconds before RunPod starts a new worker. "
            "1 = scale out immediately when a job waits 1s (lower latency, slightly more workers)."
        ),
    )

    # Webhook
    webhook_timeout_seconds: int = Field(default=10, ge=1, le=60)
    webhook_max_retries: int = Field(default=3, ge=1, le=10)

    # Logging
    log_level: LogLevel = Field(default="INFO")

    # Internal webhook (deployment-ready callback)
    internal_webhook_secret: str = Field(
        default="",
        description="Optional secret to verify internal deployment-ready requests",
    )
    internal_webhook_base_url: str = Field(
        default="",
        description="Base URL of this service for container callback (e.g. https://orch-xxx.run.app)",
    )

    # AWS / S3 (Optional, for optimized loading)
    aws_access_key_id: str = Field(default="", description="AWS Access Key ID")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Access Key")
    aws_endpoint_url: str = Field(default="", description="AWS Endpoint URL (for R2/Minio)")
    s3_model_url: str = Field(default="", description="Base S3 URL for model cache")
    shared_cache_enabled: bool = Field(
        default=True,
        description="Enable platform shared cache mode",
    )
    shared_cache_allowed_models: str = Field(
        default=(
            "stabilityai/sd-turbo,"
            "black-forest-labs/FLUX.1-schnell,"
            "stabilityai/stable-diffusion-xl-base-1.0,"
            "stabilityai/sdxl-turbo,"
            "stabilityai/stable-diffusion-3.5-large"
        ),
        description="Comma-separated model allowlist for shared cache mode",
    )
    shared_cache_reject_unlisted: bool = Field(
        default=True,
        description="Reject shared cache requests for models outside shared_cache_allowed_models",
    )

    # API
    rate_limit_requests_per_minute: int = Field(default=100, ge=1, le=1000)
    enable_endpoint_reuse: bool = Field(
        default=False,
        description="Reuse ready endpoints for same model/key instead of creating new endpoint",
    )
    stateless_mode: bool = Field(
        default=True,
        description="Avoid storing user secrets; use in-memory orchestration",
    )

    # Warm pool policy
    warm_pool_always_on_models: str = Field(
        default="",
        description="Comma-separated HF model IDs to keep warm",
    )
    warm_pool_scheduled_models: str = Field(
        default="",
        description="Comma-separated HF model IDs for scheduled warm windows",
    )
    warm_pool_schedule_hours: str = Field(
        default="09-21",
        description="Warm schedule hours in HH-HH, comma-separated windows",
    )
    warm_pool_schedule_timezone: str = Field(
        default="UTC",
        description="Timezone for scheduled warm windows",
    )

    # Log tunneling and cleanup
    log_stream_max_entries: int = Field(default=500, ge=10, le=5000)
    log_stream_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    cleanup_idle_timeout_seconds: int = Field(default=900, ge=60, le=7200)
    cleanup_failure_threshold: int = Field(default=3, ge=1, le=10)

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        u = (v or "INFO").upper()
        if u not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return u

    def resolve_secrets(self) -> None:
        """Resolve Secret Manager references for sensitive settings."""
        def _resolve(value: str) -> str:
            if not value:
                return value
            if not value.startswith("sm://"):
                return value
            secret_name = value.removeprefix("sm://")
            project = self.gcp_project_id
            try:
                from google.cloud import secretmanager
            except Exception:
                return value
            client = secretmanager.SecretManagerServiceClient()
            secret_path = f"projects/{project}/secrets/{secret_name}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": secret_path})
                return response.payload.data.decode("utf-8")
            except Exception:
                return value

        self.runpod_template_id = _resolve(self.runpod_template_id)
        self.internal_webhook_secret = _resolve(self.internal_webhook_secret)
        self.aws_access_key_id = _resolve(self.aws_access_key_id)
        self.aws_secret_access_key = _resolve(self.aws_secret_access_key)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.resolve_secrets()
    return settings
