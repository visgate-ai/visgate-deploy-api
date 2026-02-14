"""Firestore access for deployment documents."""

from datetime import UTC, datetime
from typing import Optional

from google.cloud import firestore  # type: ignore

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
    # Assuming key is the document ID for O(1) lookup
    # Alternatively, could be a field query if keys are hashed secrets
    doc = client.collection(collection).document(key).get()
    if not doc.exists:
        return None
    return doc.to_dict()
