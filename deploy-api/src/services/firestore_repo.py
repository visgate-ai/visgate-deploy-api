"""Firestore access for deployment documents and inference jobs."""

from datetime import datetime, timezone

UTC = timezone.utc

from google.cloud import firestore  # type: ignore
from google.cloud.firestore_v1.base_query import FieldFilter

from src.models.entities import DeploymentDoc, InferenceJobDoc, LogEntry


def get_firestore_client(project_id: str | None = None):  # type: ignore
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
) -> DeploymentDoc | None:
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
) -> dict | None:
    """Retrieve API key document."""
    doc = client.collection(collection).document(key).get()
    if not doc.exists:
        return None
    return doc.to_dict()


def get_gpu_registry(
    client: firestore.Client,
    collection: str = "visgate_deploy_api_gpu_registry",
) -> list[dict]:
    """Fetch list of GPU specifications from Firestore."""
    docs = client.collection(collection).stream()
    return [doc.to_dict() for doc in docs]


def get_tier_mapping(
    client: firestore.Client,
    collection: str = "visgate_deploy_api_gpu_tiers",
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


def list_deployments(
    client: firestore.Client,
    collection: str,
    user_hash: str,
    status_filter: str | None = None,
    limit: int = 20,
) -> list[DeploymentDoc]:
    """List deployments belonging to a given user (by user_hash), newest first."""
    query = client.collection(collection).where(filter=FieldFilter("user_hash", "==", user_hash))
    docs = []
    for snap in query.stream():
        data = snap.to_dict()
        if not data:
            continue
        if status_filter and data.get("status") != status_filter:
            continue
        docs.append(DeploymentDoc.from_firestore_dict(data))
    docs.sort(key=lambda doc: doc.created_at or "", reverse=True)
    return docs[: min(limit, 100)]


def find_reusable_deployment(
    client: firestore.Client,
    collection: str,
    api_key_id: str,
    hf_model_id: str,
    gpu_tier: str | None,
    user_runpod_key: str,
) -> DeploymentDoc | None:
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


def inference_job_ref(client: firestore.Client, collection: str, job_id: str):
    return client.collection(collection).document(job_id)


def get_inference_job(
    client: firestore.Client,
    collection: str,
    job_id: str,
) -> InferenceJobDoc | None:
    ref = inference_job_ref(client, collection, job_id)
    doc = ref.get()
    if not doc or not doc.exists:
        return None
    return InferenceJobDoc.from_firestore_dict(doc.to_dict())


def set_inference_job(
    client: firestore.Client,
    collection: str,
    doc: InferenceJobDoc,
) -> None:
    inference_job_ref(client, collection, doc.job_id).set(doc.to_firestore_dict())


def update_inference_job(
    client: firestore.Client,
    collection: str,
    job_id: str,
    updates: dict,
) -> None:
    inference_job_ref(client, collection, job_id).update(updates)


def list_inference_jobs(
    client: firestore.Client,
    collection: str,
    user_hash: str,
    deployment_id: str | None = None,
    limit: int = 20,
) -> list[InferenceJobDoc]:
    query = client.collection(collection).where(filter=FieldFilter("user_hash", "==", user_hash))
    docs = []
    for snap in query.stream():
        data = snap.to_dict()
        if not data:
            continue
        if deployment_id and data.get("deployment_id") != deployment_id:
            continue
        docs.append(InferenceJobDoc.from_firestore_dict(data))
    docs.sort(key=lambda doc: doc.created_at or "", reverse=True)
    return docs[: min(limit, 100)]
