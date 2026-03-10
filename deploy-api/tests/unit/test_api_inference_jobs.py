"""Unit tests for inference job lifecycle routes."""

import hashlib
import os
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from src.core.config import get_settings
from src.models.entities import DeploymentDoc
from src.services.firestore_repo import set_deployment

os.environ.setdefault("GCP_PROJECT_ID", "visgate")


def _ready_deployment_doc(runpod_key: str, *, gpu_allocated: str | None = None) -> DeploymentDoc:
    user_hash = hashlib.sha256(runpod_key.encode("utf-8")).hexdigest()
    return DeploymentDoc(
        deployment_id="dep_2026_ready123",
        status="ready",
        hf_model_id="black-forest-labs/FLUX.1-schnell",
        user_webhook_url="https://example.com/webhook",
        runpod_endpoint_id="endpoint_123",
        endpoint_url="https://api.runpod.ai/v2/endpoint_123/run",
        gpu_allocated=gpu_allocated,
        created_at="2026-03-08T10:00:00Z",
        ready_at="2026-03-08T10:01:00Z",
        user_hash=user_hash,
        provider="runpod",
        task="text_to_image",
    )


def test_create_inference_job_returns_202(
    client: TestClient,
    firestore_mock,
    auth_headers: dict,
) -> None:
    runpod_key = auth_headers["Authorization"].split(" ", 1)[1]
    set_deployment(firestore_mock, "deployments", _ready_deployment_doc(runpod_key))

    provider = Mock()
    provider.submit_job = AsyncMock(return_value={"id": "rp_job_1", "status": "IN_QUEUE", "raw_response": {}})

    with patch("src.api.routes.inference.get_provider", return_value=provider):
        resp = client.post(
            "/v1/inference/jobs",
            json={
                "deployment_id": "dep_2026_ready123",
                "task": "text_to_image",
                "input": {"prompt": "A futuristic skyline"},
            },
            headers=auth_headers,
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["deployment_id"] == "dep_2026_ready123"
    assert data["provider_job_id"] == "rp_job_1"
    assert data["status"] == "queued"
    assert data["output_destination"] == {
        "bucket_name": "platform-output",
        "endpoint_url": "https://088e0d2618d33e55a76e4d65b439d6c4.r2.cloudflarestorage.com",
        "key_prefix": f"inference/outputs/dep_2026_ready123/{data['job_id']}",
    }
    kwargs = provider.submit_job.await_args.kwargs
    assert kwargs["s3_config"]["bucketName"] == "platform-output"
    provider.submit_job.assert_awaited_once()


def test_create_inference_job_uses_request_base_for_internal_callback(
    client: TestClient,
    firestore_mock,
    auth_headers: dict,
    monkeypatch,
) -> None:
    runpod_key = auth_headers["Authorization"].split(" ", 1)[1]
    set_deployment(firestore_mock, "deployments", _ready_deployment_doc(runpod_key))
    monkeypatch.delenv("INTERNAL_WEBHOOK_BASE_URL", raising=False)
    monkeypatch.setenv("INTERNAL_WEBHOOK_SECRET", "internal-secret")
    get_settings.cache_clear()

    provider = Mock()
    provider.submit_job = AsyncMock(return_value={"id": "rp_job_1", "status": "IN_QUEUE", "raw_response": {}})

    with patch("src.api.routes.inference.get_provider", return_value=provider):
        resp = client.post(
            "/v1/inference/jobs",
            json={
                "deployment_id": "dep_2026_ready123",
                "task": "text_to_image",
                "input": {"prompt": "A futuristic skyline"},
            },
            headers=auth_headers,
        )

    assert resp.status_code == 202
    assert provider.submit_job.await_args.kwargs["webhook_url"] == (
        "http://testserver/internal/inference/jobs/"
        f"{resp.json()['job_id']}/complete?secret=internal-secret"
    )
    get_settings.cache_clear()


def test_get_inference_job_refreshes_provider_status(
    client: TestClient,
    firestore_mock,
    auth_headers: dict,
) -> None:
    runpod_key = auth_headers["Authorization"].split(" ", 1)[1]
    set_deployment(firestore_mock, "deployments", _ready_deployment_doc(runpod_key))

    provider = Mock()
    provider.submit_job = AsyncMock(return_value={"id": "rp_job_2", "status": "IN_QUEUE", "raw_response": {}})
    provider.get_job_status = AsyncMock(
        return_value={
            "id": "rp_job_2",
            "status": "COMPLETED",
            "output": {
                "image": "https://cdn.example.com/output.png",
                "artifact": {
                    "bucket_name": "user-results",
                    "endpoint_url": "https://storage.example.com",
                    "key": "jobs/output.png",
                    "url": "https://storage.example.com/user-results/jobs/output.png",
                    "content_type": "image/png",
                    "bytes": 1234,
                },
            },
            "error": None,
            "raw_response": {},
        }
    )

    with patch("src.api.routes.inference.get_provider", return_value=provider):
        create_resp = client.post(
            "/v1/inference/jobs",
            json={
                "deployment_id": "dep_2026_ready123",
                "input": {"prompt": "A futuristic skyline"},
            },
            headers=auth_headers,
        )
        job_id = create_resp.json()["job_id"]

        get_resp = client.get(f"/v1/inference/jobs/{job_id}", headers=auth_headers)

    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["status"] == "completed"
    assert data["provider_status"] == "COMPLETED"
    assert data["output"]["image"] == "https://cdn.example.com/output.png"
    assert data["artifact"]["url"] == "https://storage.example.com/user-results/jobs/output.png"
    assert data["artifact"]["content_type"] == "image/png"
    assert data["artifact"]["bytes"] == 1234
    assert data["metrics"]["queue_ms"] is None
    provider.get_job_status.assert_awaited_once()


def test_create_inference_job_without_storage_config_is_accepted(
    client: TestClient,
    firestore_mock,
    auth_headers: dict,
) -> None:
    """Platform-managed storage means callers no longer send storage config."""
    runpod_key = auth_headers["Authorization"].split(" ", 1)[1]
    set_deployment(firestore_mock, "deployments", _ready_deployment_doc(runpod_key))
    provider = Mock()
    provider.list_gpu_types = AsyncMock(return_value=[])
    provider.submit_job = AsyncMock(return_value={"id": "rp_job_nosec", "status": "IN_QUEUE", "raw_response": {}})

    with patch("src.api.routes.inference.get_provider", return_value=provider):
        resp = client.post(
            "/v1/inference/jobs",
            json={
                "deployment_id": "dep_2026_ready123",
                "input": {"prompt": "A futuristic skyline"},
            },
            headers=auth_headers,
        )

    assert resp.status_code == 202


def test_get_inference_job_estimates_cost_from_gpu_price(
    client: TestClient,
    firestore_mock,
    auth_headers: dict,
) -> None:
    runpod_key = auth_headers["Authorization"].split(" ", 1)[1]
    set_deployment(
        firestore_mock,
        "deployments",
        _ready_deployment_doc(runpod_key, gpu_allocated="AMPERE 24GB"),
    )

    provider = Mock()
    provider.list_gpu_types = AsyncMock(
        return_value=[
            {"id": "AMPERE_24", "displayName": "AMPERE 24GB", "communityPrice": 0.5, "securePrice": 0.7}
        ]
    )
    provider.submit_job = AsyncMock(return_value={"id": "rp_job_3", "status": "IN_QUEUE", "raw_response": {}})
    provider.get_job_status = AsyncMock(
        return_value={
            "id": "rp_job_3",
            "status": "COMPLETED",
            "output": {"image": "https://cdn.example.com/output.png"},
            "error": None,
            "execution_time": 60000,
            "raw_response": {},
        }
    )

    with patch("src.api.routes.inference.get_provider", return_value=provider):
        create_resp = client.post(
            "/v1/inference/jobs",
            json={
                "deployment_id": "dep_2026_ready123",
                "input": {"prompt": "A futuristic skyline"},
            },
            headers=auth_headers,
        )
        job_id = create_resp.json()["job_id"]
        get_resp = client.get(f"/v1/inference/jobs/{job_id}", headers=auth_headers)

    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["estimated_cost_usd"] == 0.008333