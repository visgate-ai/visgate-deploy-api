"""Unit/contract tests for deployment API (mocked Firestore)."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.config import get_settings

os.environ["GCP_PROJECT_ID"] = "visgate"


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
def test_post_deployments_202(
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
    deployment_create_payload: dict,
) -> None:
    """POST /v1/deployments returns 202 and creates background task."""
    resp = client.post(
        "/v1/deployments",
        json=deployment_create_payload,
        headers=auth_headers,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "deployment_id" in data
    assert data["status"] == "accepted_cold"
    assert data["model_id"] == deployment_create_payload["hf_model_id"]
    assert data["webhook_url"] == deployment_create_payload["user_webhook_url"]
    assert data["path"] == "cold"

    # Verify task was enqueued
    mock_enqueue.assert_called_once()
    args, _ = mock_enqueue.call_args
    assert args[0] == data["deployment_id"]
    assert args[1] == deployment_create_payload["user_runpod_key"]


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
def test_post_deployments_task_field_accepted(
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
) -> None:
    """POST with explicit task field returns 202 and the task is accepted."""
    resp = client.post(
        "/v1/deployments",
        json={
            "hf_model_id": "black-forest-labs/FLUX.1-schnell",
            "user_runpod_key": "rpa_xxx",
            "user_webhook_url": "https://example.com/webhook",
            "task": "text2img",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["model_id"] == "black-forest-labs/FLUX.1-schnell"
    assert "deployment_id" in data
    assert data["status"] == "accepted_cold"
    mock_enqueue.assert_called_once()


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
def test_post_deployments_canonical_task_field_accepted(
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
) -> None:
    """POST with canonical task names is accepted and normalized."""
    resp = client.post(
        "/v1/deployments",
        json={
            "hf_model_id": "black-forest-labs/FLUX.1-schnell",
            "user_runpod_key": "rpa_xxx",
            "user_webhook_url": "https://example.com/webhook",
            "task": "text_to_image",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 202
    mock_enqueue.assert_called_once()


def test_post_deployments_missing_hf_model_id_returns_422(
    client: TestClient,
    auth_headers: dict,
) -> None:
    """POST without hf_model_id returns 422 (required field)."""
    resp = client.post(
        "/v1/deployments",
        json={
            "user_runpod_key": "rpa_xxx",
            "user_webhook_url": "https://example.com/webhook",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 422


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
def test_post_deployments_allows_missing_webhook(
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
) -> None:
    resp = client.post(
        "/v1/deployments",
        json={
            "hf_model_id": "black-forest-labs/FLUX.1-schnell",
            "user_runpod_key": "rpa_xxx",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 202
    assert resp.json()["webhook_url"] == ""
    mock_enqueue.assert_called_once()


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
@patch("src.api.routes.deployments.set_deployment")
def test_post_deployments_persists_request_base_for_internal_callbacks(
    mock_set_deployment,
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERNAL_WEBHOOK_BASE_URL", raising=False)
    get_settings.cache_clear()

    resp = client.post(
        "/v1/deployments",
        json={
            "hf_model_id": "black-forest-labs/FLUX.1-schnell",
            "user_runpod_key": "rpa_xxx",
        },
        headers=auth_headers,
    )

    assert resp.status_code == 202
    stored_doc = mock_set_deployment.call_args.args[2]
    assert stored_doc.internal_webhook_base_url == "http://testserver"
    mock_enqueue.assert_called_once()
    get_settings.cache_clear()


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
@patch("src.api.routes.deployments.set_deployment")
def test_post_deployments_prefers_forwarded_https_base_for_internal_callbacks(
    mock_set_deployment,
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERNAL_WEBHOOK_BASE_URL", raising=False)
    get_settings.cache_clear()

    resp = client.post(
        "/v1/deployments",
        json={
            "hf_model_id": "black-forest-labs/FLUX.1-schnell",
            "user_runpod_key": "rpa_xxx",
        },
        headers={
            **auth_headers,
            "x-forwarded-proto": "https",
            "x-forwarded-host": "visgate-deploy-api.example.com",
        },
    )

    assert resp.status_code == 202
    stored_doc = mock_set_deployment.call_args.args[2]
    assert stored_doc.internal_webhook_base_url == "https://visgate-deploy-api.example.com"
    mock_enqueue.assert_called_once()
    get_settings.cache_clear()


def test_post_deployments_401_without_auth(client: TestClient, deployment_create_payload: dict) -> None:
    """POST without Bearer returns 401."""
    resp = client.post("/v1/deployments", json=deployment_create_payload)
    assert resp.status_code == 401


def test_health(client: TestClient) -> None:
    """GET /health returns 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root(client: TestClient) -> None:
    """GET / returns service info."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "visgate-deploy-api" in resp.json().get("service", "")


def test_private_fields_require_private_scope(
    client: TestClient,
    auth_headers: dict,
    deployment_create_payload: dict,
) -> None:
    """Private S3 fields must not be accepted unless cache_scope=private."""
    payload = {
        **deployment_create_payload,
        "cache_scope": "off",
        "user_s3_url": "s3://example/models",
    }
    resp = client.post("/v1/deployments", json=payload, headers=auth_headers)
    assert resp.status_code == 400
    assert "cache_scope=private" in resp.json()["message"]


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
def test_shared_cache_rejects_unlisted_model(
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
    deployment_create_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared cache should reject models outside allowlist when strict mode is enabled."""
    monkeypatch.setenv("S3_MODEL_URL", "s3://platform-cache/models")
    monkeypatch.setenv("SHARED_CACHE_ALLOWED_MODELS", "stabilityai/sd-turbo")
    monkeypatch.setenv("SHARED_CACHE_REJECT_UNLISTED", "true")

    from src.core.config import get_settings

    get_settings.cache_clear()
    payload = {
        **deployment_create_payload,
        "hf_model_id": "black-forest-labs/FLUX.1-schnell",
        "cache_scope": "shared",
    }
    resp = client.post("/v1/deployments", json=payload, headers=auth_headers)
    get_settings.cache_clear()

    assert resp.status_code == 400
    assert "allowlisted popular models" in resp.json()["message"]
    mock_enqueue.assert_not_called()
