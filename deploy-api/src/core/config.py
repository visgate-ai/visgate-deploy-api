"""Application settings loaded from environment with validation."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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
    gcp_project_id: str = Field(default="", description="GCP project ID (leave empty for local dev with in-memory storage)")
    gcp_location: str = Field(default="europe-west3", description="GCP region for Cloud Tasks")
    cloud_tasks_queue_path: str = Field(
        default="",
        description="Full path to Cloud Tasks queue (projects/.../locations/.../queues/...)",
    )
    cloud_tasks_service_account: str = Field(
        default="",
        description=(
            "Service account email attached as OIDC token to Cloud Tasks HTTP requests. "
            "The SA must have roles/run.invoker on the Cloud Run service. "
            "Example: visgate-sa@visgate.iam.gserviceaccount.com"
        ),
    )
    use_memory_repo: bool = Field(
        default=False,
        description="Use in-memory storage instead of Firestore. Auto-enabled when GCP_PROJECT_ID is empty.",
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
        default="visgate_deploy_api_api_keys",
        min_length=1,
        description="Firestore collection for API keys",
    )
    firestore_collection_gpu_registry: str = Field(
        default="visgate_deploy_api_gpu_registry",
        min_length=1,
        description="Firestore collection for GPU specifications",
    )
    firestore_collection_gpu_tiers: str = Field(
        default="visgate_deploy_api_gpu_tiers",
        min_length=1,
        description="Firestore collection for GPU tier mappings",
    )

    # Runpod
    docker_image: str = Field(
        default="visgateai/inference:latest",
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

    # Platform HF token (used when user does not supply their own hf_token)
    hf_pro_access_token: str = Field(
        default="",
        validation_alias=AliasChoices("visgate_deploy_api_hf_pro_access_token", "hf_pro_access_token"),
        description="Platform HF Pro token — auto-injected for gated models when caller omits hf_token",
    )

    # ── Cloudflare R2 — Read/Write key ────────────────────────────────────────
    # Used by the Deploy API only (manifest writes, HF→R2 cache uploads).
    # NEVER injected into RunPod workers — workers only get the read-only key.
    r2_access_key_id_rw: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deployapi_r2_access_key_id_rw",   # new canonical name
            "visgate_deploy_api_r2_access_key_id_rw",  # GCP secret name (legacy)
            "aws_access_key_id",                        # backward compat
        ),
        description="Cloudflare R2 RW Access Key ID — Deploy API only. Never sent to RunPod workers.",
    )
    r2_secret_access_key_rw: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deployapi_r2_secret_access_key_rw",
            "visgate_deploy_api_r2_secret_access_key_rw",
            "aws_secret_access_key",
        ),
        description="Cloudflare R2 RW Secret Access Key — Deploy API only.",
    )
    r2_endpoint_url: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deployapi_r2_endpoint_url",
            "visgate_deploy_api_r2_endpoint_url",
            "visgate_deploy_api_s3_api_r2",  # GCP secret name (legacy)
            "aws_endpoint_url",
        ),
        description="Cloudflare R2 endpoint URL — e.g. https://ACCOUNT.r2.cloudflarestorage.com",
    )
    r2_model_base_url: str = Field(
        default="s3://visgate-models/models",
        validation_alias=AliasChoices(
            "visgate_deployapi_r2_model_base_url",
            "visgate_deploy_api_s3_model_url_r2",
            "s3_model_url",
        ),
        description="Base S3 path for the platform R2 model cache (e.g. s3://visgate-models/models)",
    )

    # ── Cloudflare R2 — Read-only key ─────────────────────────────────────────
    # Injected into RunPod workers so they can sync models from R2.
    # Workers can only read — they cannot write or delete objects.
    r2_access_key_id_ro: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deployapi_r2_access_key_id_ro",
            "visgate_deploy_api_r2_access_key_id_ro",
            "visgate_deploy_api_r2_access_key_id_r",  # GCP secret name (legacy)
            "r2_access_key_id_r",
        ),
        description="Cloudflare R2 read-only Access Key ID — injected into RunPod workers for shared cache reads.",
    )
    r2_secret_access_key_ro: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deployapi_r2_secret_access_key_ro",
            "visgate_deploy_api_r2_secret_access_key_ro",
            "visgate_deploy_api_r2_secret_access_key_r",
            "r2_secret_access_key_r",
        ),
        description="Cloudflare R2 read-only Secret Access Key — injected into RunPod workers.",
    )

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
    cache_manifest_ttl_seconds: int = Field(
        default=600,
        ge=0,
        le=3600,
        description="In-memory TTL for the R2 manifest cache to reduce repeated object reads.",
    )
    cache_model_task_timeout_seconds: int = Field(
        default=3600,
        ge=60,
        le=7200,
        description="Maximum allowed runtime for a single cache-model task attempt.",
    )
    cache_model_task_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for cache-model tasks before surfacing failure to Cloud Tasks.",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        u = (v or "INFO").upper()
        if u not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return u

    @property
    def effective_use_memory_repo(self) -> bool:
        """True when memory repo should be used (explicit flag or no GCP project set)."""
        return self.use_memory_repo or not self.gcp_project_id

    def resolve_secrets(self) -> None:
        """Resolve Secret Manager references for sensitive settings."""
        def _resolve(value: str) -> str:
            if not value:
                return value
            if not value.startswith("sm://"):
                return value.strip()
            secret_name = value.removeprefix("sm://")
            project = self.gcp_project_id
            try:
                from google.cloud import secretmanager
            except Exception:
                return value.strip()
            client = secretmanager.SecretManagerServiceClient()
            secret_path = f"projects/{project}/secrets/{secret_name}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": secret_path})
                return response.payload.data.decode("utf-8").strip()
            except Exception:
                return value.strip()

        self.runpod_template_id = _resolve(self.runpod_template_id)
        self.internal_webhook_secret = _resolve(self.internal_webhook_secret)
        self.hf_pro_access_token = _resolve(self.hf_pro_access_token)
        self.r2_access_key_id_rw = _resolve(self.r2_access_key_id_rw)
        self.r2_secret_access_key_rw = _resolve(self.r2_secret_access_key_rw)
        self.r2_access_key_id_ro = _resolve(self.r2_access_key_id_ro)
        self.r2_secret_access_key_ro = _resolve(self.r2_secret_access_key_ro)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.resolve_secrets()
    return settings
