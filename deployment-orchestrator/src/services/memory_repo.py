"""In-memory repository for deployments (non-persistent)."""
from datetime import UTC, datetime
from typing import Optional, Any
from src.models.entities import DeploymentDoc, LogEntry

# Global store
_deployments: dict[str, dict] = {}
_api_keys: dict[str, dict] = {}

# Just to mock the signature
class MemoryClient:
    pass

def get_firestore_client(project_id: Optional[str] = None) -> MemoryClient:
    return MemoryClient()

def get_deployment(
    client: Any,
    collection: str,
    deployment_id: str,
) -> Optional[DeploymentDoc]:
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
) -> Optional[dict]:
    # Hardcoded or env var based for simplicity if needed
    import os
    if key == os.getenv("ORCHESTRATOR_API_KEY", "test-key"):
        return {"active": True, "owner": "admin"}
    return None
