"""Firestore document and in-memory entity models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogEntry:
    """Single log entry stored in Firestore deployment doc."""

    timestamp: str  # ISO format
    level: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "level": self.level, "message": self.message}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogEntry":
        return cls(
            timestamp=d.get("timestamp", ""),
            level=d.get("level", "INFO"),
            message=d.get("message", ""),
        )


@dataclass
class DeploymentDoc:
    """Firestore deployment document shape."""

    deployment_id: str
    status: str
    hf_model_id: str
    user_webhook_url: str | None
    user_runpod_key_ref: str | None = None  # Deprecated: avoid storing user secrets
    gpu_tier: str | None = None
    hf_token_ref: str | None = None  # Deprecated: avoid storing user secrets
    runpod_endpoint_id: str | None = None
    endpoint_url: str | None = None
    gpu_allocated: str | None = None
    model_vram_gb: int | None = None
    logs: list[LogEntry] = field(default_factory=list)
    error: str | None = None
    created_at: str = ""
    ready_at: str | None = None
    api_key_id: str | None = None
    user_hash: str | None = None
    provider: str | None = None
    endpoint_name: str | None = None
    pool_policy: str | None = None
    region: str | None = None
    task: str | None = None

    def to_firestore_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "status": self.status,
            "hf_model_id": self.hf_model_id,
            "user_runpod_key_ref": self.user_runpod_key_ref,
            "user_webhook_url": self.user_webhook_url,
            "gpu_tier": self.gpu_tier,
            "hf_token_ref": self.hf_token_ref,
            "runpod_endpoint_id": self.runpod_endpoint_id,
            "endpoint_url": self.endpoint_url,
            "gpu_allocated": self.gpu_allocated,
            "model_vram_gb": self.model_vram_gb,
            "logs": [e.to_dict() for e in self.logs],
            "error": self.error,
            "created_at": self.created_at,
            "ready_at": self.ready_at,
            "api_key_id": self.api_key_id,
            "user_hash": self.user_hash,
            "provider": self.provider,
            "endpoint_name": self.endpoint_name,
            "pool_policy": self.pool_policy,
            "region": self.region,
            "task": self.task,
        }

    @classmethod
    def from_firestore_dict(cls, d: dict[str, Any]) -> "DeploymentDoc":
        logs = [LogEntry.from_dict(e) for e in d.get("logs", [])]
        return cls(
            deployment_id=d.get("deployment_id", ""),
            status=d.get("status", "validating"),
            hf_model_id=d.get("hf_model_id", ""),
            user_runpod_key_ref=d.get("user_runpod_key_ref"),
            user_webhook_url=d.get("user_webhook_url"),
            gpu_tier=d.get("gpu_tier"),
            hf_token_ref=d.get("hf_token_ref"),
            runpod_endpoint_id=d.get("runpod_endpoint_id"),
            endpoint_url=d.get("endpoint_url"),
            gpu_allocated=d.get("gpu_allocated"),
            model_vram_gb=d.get("model_vram_gb"),
            logs=logs,
            error=d.get("error"),
            created_at=d.get("created_at", ""),
            ready_at=d.get("ready_at"),
            api_key_id=d.get("api_key_id"),
            user_hash=d.get("user_hash"),
            provider=d.get("provider"),
            endpoint_name=d.get("endpoint_name"),
            pool_policy=d.get("pool_policy"),
            region=d.get("region"),
            task=d.get("task"),
        )


@dataclass
class InferenceJobDoc:
    """Firestore inference job document shape."""

    job_id: str
    deployment_id: str
    provider: str
    provider_job_id: str
    endpoint_url: str
    status: str
    provider_status: str
    gpu_allocated: str | None = None
    gpu_price_per_hour_usd: float | None = None
    hf_model_id: str = ""
    task: str | None = None
    input_payload: dict[str, Any] = field(default_factory=dict)
    output_destination: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    estimated_cost_usd: float | None = None
    output_payload: Any = None
    output_preview: Any = None
    error: Any = None
    progress: Any = None
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    user_hash: str | None = None
    user_webhook_url: str | None = None
    webhook_delivered_at: str | None = None

    def to_firestore_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "deployment_id": self.deployment_id,
            "provider": self.provider,
            "provider_job_id": self.provider_job_id,
            "endpoint_url": self.endpoint_url,
            "status": self.status,
            "provider_status": self.provider_status,
            "gpu_allocated": self.gpu_allocated,
            "gpu_price_per_hour_usd": self.gpu_price_per_hour_usd,
            "hf_model_id": self.hf_model_id,
            "task": self.task,
            "input_payload": self.input_payload,
            "output_destination": self.output_destination,
            "artifact": self.artifact,
            "metrics": self.metrics,
            "estimated_cost_usd": self.estimated_cost_usd,
            "output_payload": self.output_payload,
            "output_preview": self.output_preview,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "user_hash": self.user_hash,
            "user_webhook_url": self.user_webhook_url,
            "webhook_delivered_at": self.webhook_delivered_at,
        }

    @classmethod
    def from_firestore_dict(cls, d: dict[str, Any]) -> "InferenceJobDoc":
        return cls(
            job_id=d.get("job_id", ""),
            deployment_id=d.get("deployment_id", ""),
            provider=d.get("provider", "runpod"),
            provider_job_id=d.get("provider_job_id", ""),
            endpoint_url=d.get("endpoint_url", ""),
            status=d.get("status", "queued"),
            provider_status=d.get("provider_status", "IN_QUEUE"),
            gpu_allocated=d.get("gpu_allocated"),
            gpu_price_per_hour_usd=d.get("gpu_price_per_hour_usd"),
            hf_model_id=d.get("hf_model_id", ""),
            task=d.get("task"),
            input_payload=d.get("input_payload") or {},
            output_destination=d.get("output_destination"),
            artifact=d.get("artifact"),
            metrics=d.get("metrics"),
            estimated_cost_usd=d.get("estimated_cost_usd"),
            output_payload=d.get("output_payload"),
            output_preview=d.get("output_preview"),
            error=d.get("error"),
            progress=d.get("progress"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            completed_at=d.get("completed_at"),
            user_hash=d.get("user_hash"),
            user_webhook_url=d.get("user_webhook_url"),
            webhook_delivered_at=d.get("webhook_delivered_at"),
        )
