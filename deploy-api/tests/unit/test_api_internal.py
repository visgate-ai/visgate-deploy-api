"""Unit tests for internal callback routes."""

import hashlib

from fastapi.testclient import TestClient

from src.core.config import get_settings
from src.models.entities import DeploymentDoc, InferenceJobDoc
from src.services.firestore_repo import set_deployment, set_inference_job


def _deployment_doc(runpod_key: str) -> DeploymentDoc:
    user_hash = hashlib.sha256(runpod_key.encode("utf-8")).hexdigest()
    return DeploymentDoc(
        deployment_id="dep_internal_1",
        status="ready",
        hf_model_id="black-forest-labs/FLUX.1-schnell",
        user_webhook_url="https://example.com/webhook",
        runpod_endpoint_id="endpoint_123",
        endpoint_url="https://api.runpod.ai/v2/endpoint_123/run",
        gpu_allocated="A40",
        created_at="2026-03-08T10:00:00Z",
        ready_at="2026-03-08T10:01:00Z",
        user_hash=user_hash,
        provider="runpod",
        task="text_to_image",
    )


def _job_doc() -> InferenceJobDoc:
    return InferenceJobDoc(
        job_id="job_internal_1",
        deployment_id="dep_internal_1",
        provider="runpod",
        provider_job_id="rp_job_1",
        endpoint_url="https://api.runpod.ai/v2/endpoint_123/run",
        status="queued",
        provider_status="IN_QUEUE",
        task="text_to_image",
        input_payload={"prompt": "A city at sunrise"},
        output_destination={
            "bucket_name": "user-results",
            "endpoint_url": "https://storage.example.com",
            "key_prefix": "jobs/dep_internal_1",
        },
        created_at="2026-03-08T10:02:00Z",
        updated_at="2026-03-08T10:02:00Z",
    )


def test_model_cached_rejects_query_secret_when_header_missing(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setenv("INTERNAL_WEBHOOK_SECRET", "internal-secret")
    get_settings.cache_clear()

    resp = client.post(
        "/internal/model-cached?secret=internal-secret",
        json={"hf_model_id": "black-forest-labs/FLUX.1-schnell", "deployment_id": "dep_internal_1"},
    )

    assert resp.status_code == 403
    assert resp.json()["detail"] == "Invalid internal secret"
    get_settings.cache_clear()


def test_inference_job_complete_rejects_deployment_mismatch(
    client: TestClient,
    firestore_mock,
    auth_headers: dict,
    monkeypatch,
) -> None:
    runpod_key = auth_headers["Authorization"].split(" ", 1)[1]
    monkeypatch.setenv("INTERNAL_WEBHOOK_SECRET", "internal-secret")
    get_settings.cache_clear()
    settings = get_settings()

    set_deployment(firestore_mock, settings.firestore_collection_deployments, _deployment_doc(runpod_key))
    set_inference_job(firestore_mock, settings.firestore_collection_inference_jobs, _job_doc())

    resp = client.post(
        "/internal/inference/jobs/job_internal_1/complete",
        json={"status": "COMPLETED", "deployment_id": "dep_other"},
        headers={"X-Visgate-Internal-Secret": "internal-secret"},
    )

    assert resp.status_code == 403
    assert resp.json()["detail"] == "Deployment mismatch for job callback"
    get_settings.cache_clear()
