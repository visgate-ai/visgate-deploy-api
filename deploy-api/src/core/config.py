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
        default="visgateai/inference:latest",
        min_length=1,
        description="Docker image for Runpod inference workers",
    )
    runpod_template_id: str = Field(
        default="",
        description="Runpod template ID for the inference image (create in Runpod console)",
    )
    runpod_max_retries: int = Field(default=3, ge=1, le=10)
    runpod_graphql_url: str = Field(
        default="https://api.runpod.io/graphql",
        description="Runpod GraphQL API URL",
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

    # API
    rate_limit_requests_per_minute: int = Field(default=100, ge=1, le=1000)

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        u = (v or "INFO").upper()
        if u not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return u


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
