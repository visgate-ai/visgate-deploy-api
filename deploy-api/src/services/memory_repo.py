"""In-memory repository for deployments and inference jobs (non-persistent)."""
import os
from datetime import UTC, datetime
from typing import Any

from src.models.entities import DeploymentDoc, InferenceJobDoc
from src.services.gpu_registry import DEFAULT_GPU_REGISTRY, DEFAULT_TIER_MAPPING

# Global store
_deployments: dict[str, dict] = {}
_inference_jobs: dict[str, dict] = {}
_api_keys: dict[str, dict] = {}

# Just to mock the signature
class MemoryClient:
    pass

def get_firestore_client(project_id: str | None = None) -> MemoryClient:
    return MemoryClient()

def get_deployment(
    client: Any,
    collection: str,
    deployment_id: str,
) -> DeploymentDoc | None:
    data = _deployments.get(deployment_id)
    if not data:
        return None
    return DeploymentDoc.from_firestore_dict(data)

def set_deployment(
    client: Any,
    collection: str,
    doc: DeploymentDoc,
) -> None:
    _deployments[doc.deployment_id] = doc.to_firestore_dict()

def update_deployment(
    client: Any,
    collection: str,
    deployment_id: str,
    updates: dict,
) -> None:
    if deployment_id in _deployments:
        _deployments[deployment_id].update(updates)

def append_log(
    client: Any,
    collection: str,
    deployment_id: str,
    level: str,
    message: str,
) -> None:
    if deployment_id in _deployments:
        entry = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": level,
            "message": message
        }
        logs = _deployments[deployment_id].get("logs", [])
        logs.append(entry)
        _deployments[deployment_id]["logs"] = logs

def get_api_key(
    client: Any,
    collection: str,
    key: str,
) -> dict | None:
    """Accept any non-empty key in local dev mode (or match ORCHESTRATOR_API_KEY env var)."""
    expected = os.getenv("ORCHESTRATOR_API_KEY", "")
    if expected:
        return {"active": True, "owner": "admin"} if key == expected else None
    # No key configured — accept any non-empty value for local dev
    return {"active": True, "owner": "admin"} if key else None


def get_gpu_registry(
    client: Any,
    collection: str,
) -> list:
    """Return default GPU registry (no Firestore needed)."""
    return list(DEFAULT_GPU_REGISTRY)


def get_tier_mapping(
    client: Any,
    collection: str,
) -> dict:
    """Return default tier mapping (no Firestore needed)."""
    return dict(DEFAULT_TIER_MAPPING)


def list_deployments(
    client: Any,
    collection: str,
    user_hash: str,
    status_filter: str | None = None,
    limit: int = 20,
) -> list:
    """Return deployments for given user_hash from in-memory store."""
    results = [
        DeploymentDoc.from_firestore_dict(data)
        for data in _deployments.values()
        if data.get("user_hash") == user_hash
        and (not status_filter or data.get("status") == status_filter)
    ]
    results.sort(key=lambda d: d.created_at or "", reverse=True)
    return results[:min(limit, 100)]


def find_reusable_deployment(
    client: Any,
    collection: str,
    api_key_id: str,
    hf_model_id: str,
    gpu_tier: str | None,
    user_runpod_key: str,
) -> None:
    """In-memory mode does not support endpoint reuse — always returns None."""
    return None


def get_inference_job(
    client: Any,
    collection: str,
    job_id: str,
) -> InferenceJobDoc | None:
    data = _inference_jobs.get(job_id)
    if not data:
        return None
    return InferenceJobDoc.from_firestore_dict(data)


def set_inference_job(
    client: Any,
    collection: str,
    doc: InferenceJobDoc,
) -> None:
    _inference_jobs[doc.job_id] = doc.to_firestore_dict()


def update_inference_job(
    client: Any,
    collection: str,
    job_id: str,
    updates: dict,
) -> None:
    if job_id in _inference_jobs:
        _inference_jobs[job_id].update(updates)


def list_inference_jobs(
    client: Any,
    collection: str,
    user_hash: str,
    deployment_id: str | None = None,
    limit: int = 20,
) -> list[InferenceJobDoc]:
    results = [
        InferenceJobDoc.from_firestore_dict(data)
        for data in _inference_jobs.values()
        if data.get("user_hash") == user_hash and (not deployment_id or data.get("deployment_id") == deployment_id)
    ]
    results.sort(key=lambda d: d.created_at or "", reverse=True)
    return results[:min(limit, 100)]
