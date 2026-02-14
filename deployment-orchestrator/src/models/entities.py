"""Firestore document and in-memory entity models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


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
    user_runpod_key_ref: str  # Secret Manager path or encrypted ref
    user_webhook_url: str
    gpu_tier: Optional[str] = None
    hf_token_ref: Optional[str] = None
    runpod_endpoint_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    gpu_allocated: Optional[str] = None
    model_vram_gb: Optional[int] = None
    logs: list[LogEntry] = field(default_factory=list)
    error: Optional[str] = None
    created_at: str = ""
    ready_at: Optional[str] = None
    api_key_id: Optional[str] = None

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
        }

    @classmethod
    def from_firestore_dict(cls, d: dict[str, Any]) -> "DeploymentDoc":
        logs = [LogEntry.from_dict(e) for e in d.get("logs", [])]
        return cls(
            deployment_id=d.get("deployment_id", ""),
            status=d.get("status", "validating"),
            hf_model_id=d.get("hf_model_id", ""),
            user_runpod_key_ref=d.get("user_runpod_key_ref", ""),
            user_webhook_url=d.get("user_webhook_url", ""),
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
        )
