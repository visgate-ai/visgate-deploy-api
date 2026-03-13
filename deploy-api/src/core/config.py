"""Application settings loaded from environment with validation."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Legacy secret/env names are explicitly forbidden to prevent accidental fallback
# to historical credentials. Service startup will fail if any of these are set.
LEGACY_SECRET_ENV_NAMES = {
    "VISGATE_DEPLOYAPI_R2_ACCESS_KEY_ID_RW",
    "VISGATE_DEPLOYAPI_R2_SECRET_ACCESS_KEY_RW",
    "VISGATE_DEPLOYAPI_R2_ENDPOINT_URL",
    "VISGATE_DEPLOYAPI_R2_ACCESS_KEY_ID_RO",
    "VISGATE_DEPLOYAPI_R2_SECRET_ACCESS_KEY_RO",
    "VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_RW",
    "VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_RW",
    "VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_R",
    "VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_R",
    "VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_RO",
    "VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_RO",
    "VISGATE_DEPLOY_API_S3_API_R2",
    "VISGATE_DEPLOY_API_R2_ENDPOINT_URL",
    "VISGATE_DEPLOY_API_INFERENCE_R2_BUCKET_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
}


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
    root_path: str = Field(
        default="",
        validation_alias=AliasChoices("visgate_deploy_api_root_path", "root_path"),
        description="Optional public URL prefix when the service is published behind a reverse proxy path.",
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
    firestore_collection_inference_jobs: str = Field(
        default="inference_jobs",
        min_length=1,
        description="Firestore collection for inference job documents",
    )

    # Runpod
    inference_provider: str = Field(
        default="runpod",
        description="Active inference backend: runpod or local.",
    )
    docker_image: str = Field(
        default="visgateai/inference:latest",
        min_length=1,
        description="Docker image for Runpod inference workers",
    )
    docker_image_image: str = Field(
        default="visgateai/inference-image:latest",
        min_length=1,
        description="Docker image for image-focused Runpod workers",
    )
    docker_image_audio: str = Field(
        default="visgateai/inference-audio:latest",
        min_length=1,
        description="Docker image for audio-focused Runpod workers",
    )
    docker_image_video: str = Field(
        default="visgateai/inference-video:latest",
        min_length=1,
        description="Docker image for video-focused Runpod workers",
    )
    runpod_template_id: str = Field(
        default="",
        description="Runpod template ID for the inference image (create in Runpod console)",
    )
    runpod_template_id_image: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_runpod_template_id_image",
            "runpod_template_id_image",
        ),
        description="Runpod template ID for image worker deployments",
    )
    runpod_template_id_audio: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_runpod_template_id_audio",
            "runpod_template_id_audio",
        ),
        description="Runpod template ID for audio worker deployments",
    )
    runpod_template_id_video: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_runpod_template_id_video",
            "runpod_template_id_video",
        ),
        description="Runpod template ID for video worker deployments",
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
        default=1,
        ge=1,
        le=10,
        description=(
            "Minimum warm workers per endpoint. "
            "1 = one worker starts during deployment so readiness can be observed before accepting jobs. "
            "Higher values trade extra idle GPU cost for lower burst latency."
        ),
    )
    runpod_workers_min_video: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Minimum warm workers for video endpoints to avoid first-job cold starts after deployment readiness.",
    )
    runpod_workers_max: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Maximum concurrent workers per endpoint",
    )
    runpod_idle_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description=(
            "Seconds a worker stays alive after its last job. "
            "300s is a good balance: absorbs burst traffic without burning too many idle GPU hours. "
            "Lower = cheaper but more cold starts; higher = faster burst but more idle cost."
        ),
    )
    runpod_idle_timeout_seconds_video: int = Field(
        default=600,
        ge=30,
        le=3600,
        description="Idle timeout for video workers in seconds to avoid rapid warm-worker eviction.",
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
    runpod_execution_timeout_ms: int = Field(
        default=300000,
        ge=5000,
        le=604800000,
        description="Default RunPod execution timeout for endpoint jobs in milliseconds",
    )
    runpod_execution_timeout_ms_video: int = Field(
        default=900000,
        ge=5000,
        le=604800000,
        description="RunPod execution timeout for video endpoint jobs in milliseconds",
    )
    local_worker_url_image: str = Field(
        default="http://127.0.0.1:8101",
        description="Base URL for the local image worker when inference_provider=local.",
    )
    local_worker_url_audio: str = Field(
        default="http://127.0.0.1:8102",
        description="Base URL for the local audio worker when inference_provider=local.",
    )
    local_worker_url_video: str = Field(
        default="http://127.0.0.1:8103",
        description="Base URL for the local video worker when inference_provider=local.",
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

    # Smoke-test credentials
    smoke_test_runpod_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_smoke_test_runpod",
            "smoke_test_runpod_api_key",
        ),
        description="Optional RunPod API key used by live smoke-test automation.",
    )
    smoke_test_hf_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_smoke_test_hf_key",
            "smoke_test_hf_key",
        ),
        description="Optional Hugging Face token used by smoke tests for gated models.",
    )

    # ── Cloudflare R2 — Read/Write key ────────────────────────────────────────
    # Used by the Deploy API only (manifest writes, HF→R2 cache uploads).
    # NEVER injected into RunPod workers — workers only get the read-only key.
    r2_access_key_id_rw: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_inference_r2_access_key_id_output_rw",
        ),
        description="Cloudflare R2 RW Access Key ID (output RW) — Deploy API only.",
    )
    r2_secret_access_key_rw: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_inference_r2_secret_access_key_output_rw",
        ),
        description="Cloudflare R2 RW Secret Access Key (output RW) — Deploy API only.",
    )
    r2_endpoint_url: str = Field(
        default="https://088e0d2618d33e55a76e4d65b439d6c4.r2.cloudflarestorage.com",
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
            "visgate_deploy_api_inference_r2_access_key_id_input_r",
        ),
        description="Cloudflare R2 read-only Access Key ID (input R) — injected into RunPod workers.",
    )
    r2_secret_access_key_ro: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_inference_r2_secret_access_key_input_r",
        ),
        description="Cloudflare R2 read-only Secret Access Key (input R) — injected into RunPod workers.",
    )
    inference_r2_bucket_name_output: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_inference_r2_bucket_name_output",
        ),
        description="Optional default output bucket name for inference smoke tests.",
    )
    inference_r2_bucket_name_input: str = Field(
        default="",
        validation_alias=AliasChoices(
            "visgate_deploy_api_inference_r2_bucket_name_input",
        ),
        description="Optional input bucket name identifier for platform configuration.",
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
            "stabilityai/stable-diffusion-3.5-large,"
            "openai/whisper-large-v3,"
            "Wan-AI/Wan2.1-T2V-1.3B,"
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
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
        default=True,
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

    @field_validator("inference_provider", mode="before")
    @classmethod
    def validate_inference_provider(cls, value: str | None) -> str:
        normalized = (value or "runpod").strip().lower()
        if normalized not in {"runpod", "local"}:
            raise ValueError("inference_provider must be one of ('runpod', 'local')")
        return normalized

    def resolve_secrets(self) -> None:
        """Resolve Secret Manager references for sensitive settings."""
        legacy_present = sorted(name for name in LEGACY_SECRET_ENV_NAMES if os.getenv(name))
        if legacy_present:
            raise ValueError(
                "Legacy secret/env names are not allowed. Remove these keys and use canonical names only: "
                + ", ".join(legacy_present)
            )

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
        self.runpod_template_id_image = _resolve(self.runpod_template_id_image)
        self.runpod_template_id_audio = _resolve(self.runpod_template_id_audio)
        self.runpod_template_id_video = _resolve(self.runpod_template_id_video)
        self.internal_webhook_secret = _resolve(self.internal_webhook_secret)
        self.smoke_test_runpod_api_key = _resolve(self.smoke_test_runpod_api_key)
        self.smoke_test_hf_key = _resolve(self.smoke_test_hf_key)
        self.r2_access_key_id_rw = _resolve(self.r2_access_key_id_rw)
        self.r2_secret_access_key_rw = _resolve(self.r2_secret_access_key_rw)
        self.r2_access_key_id_ro = _resolve(self.r2_access_key_id_ro)
        self.r2_secret_access_key_ro = _resolve(self.r2_secret_access_key_ro)
        self.inference_r2_bucket_name_output = _resolve(self.inference_r2_bucket_name_output)
        self.inference_r2_bucket_name_input = _resolve(self.inference_r2_bucket_name_input)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.resolve_secrets()
    return settings
