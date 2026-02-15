"""Firestore access for deployment documents."""

from datetime import UTC, datetime
from typing import Optional

from google.cloud import firestore  # type: ignore
from google.cloud.firestore_v1.base_query import FieldFilter

from src.models.entities import DeploymentDoc, LogEntry


def get_firestore_client(project_id: Optional[str] = None):  # type: ignore
    """Return Firestore client (singleton optional)."""
    if project_id:
        return firestore.Client(project=project_id)
    return firestore.Client()


def deployment_ref(client: firestore.Client, collection: str, deployment_id: str):
    """Return document reference for a deployment."""
    return client.collection(collection).document(deployment_id)


def get_deployment(
    client: firestore.Client,
    collection: str,
    deployment_id: str,
) -> Optional[DeploymentDoc]:
    """Load deployment document by ID."""
    ref = deployment_ref(client, collection, deployment_id)
    doc = ref.get()
    if not doc or not doc.exists:
        return None
    return DeploymentDoc.from_firestore_dict(doc.to_dict())


def set_deployment(
    client: firestore.Client,
    collection: str,
    doc: DeploymentDoc,
) -> None:
    """Create or overwrite deployment document."""
    ref = deployment_ref(client, collection, doc.deployment_id)
    ref.set(doc.to_firestore_dict())


def update_deployment(
    client: firestore.Client,
    collection: str,
    deployment_id: str,
    updates: dict,
) -> None:
    """Merge updates into deployment document."""
    ref = deployment_ref(client, collection, deployment_id)
    ref.update(updates)


def append_log(
    client: firestore.Client,
    collection: str,
    deployment_id: str,
    level: str,
    message: str,
) -> None:
    """Append a log entry to deployment doc."""
    ref = deployment_ref(client, collection, deployment_id)
    entry = LogEntry(
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        level=level,
        message=message,
    )
    ref.update({"logs": firestore.ArrayUnion([entry.to_dict()])})


def get_api_key(
    client: firestore.Client,
    collection: str,
    key: str,
) -> Optional[dict]:
    """Retrieve API key document."""
    doc = client.collection(collection).document(key).get()
    if not doc.exists:
        return None
    return doc.to_dict()


def get_gpu_registry(
    client: firestore.Client,
    collection: str = "gpu_registry",
) -> list[dict]:
    """Fetch list of GPU specifications from Firestore."""
    docs = client.collection(collection).stream()
    return [doc.to_dict() for doc in docs]


def get_tier_mapping(
    client: firestore.Client,
    collection: str = "gpu_tiers",
) -> dict[str, list[str]]:
    """Fetch tier to GPU ID mappings from Firestore."""
    docs = client.collection(collection).stream()
    mapping = {}
    for doc in docs:
        data = doc.to_dict()
        name = doc.id.upper()
        ids = data.get("gpu_ids", [])
        mapping[name] = ids
    return mapping


def find_reusable_deployment(
    client: firestore.Client,
    collection: str,
    api_key_id: str,
    hf_model_id: str,
    gpu_tier: Optional[str],
    user_runpod_key: str,
) -> Optional[DeploymentDoc]:
    """
    Find an active deployment for the same caller/model/key so we can reuse endpoint.
    """
    active_statuses = {"ready", "creating_endpoint", "downloading_model", "loading_model"}
    query = client.collection(collection).where(
        filter=FieldFilter("api_key_id", "==", api_key_id)
    ).limit(50)
    for snap in query.stream():
        data = snap.to_dict() or {}
        if data.get("hf_model_id") != hf_model_id:
            continue
        if data.get("gpu_tier") != gpu_tier:
            continue
        if data.get("user_runpod_key_ref") != user_runpod_key:
            continue
        if data.get("status") not in active_statuses:
            continue
        if not data.get("endpoint_url") or not data.get("runpod_endpoint_id"):
            continue
        return DeploymentDoc.from_firestore_dict(data)
    return None
