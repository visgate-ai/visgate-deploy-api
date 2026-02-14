"""Unit/contract tests for deployment API (mocked Firestore)."""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

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
    assert data["status"] == "validating"
    assert data["model_id"] == deployment_create_payload["hf_model_id"]
    assert data["webhook_url"] == deployment_create_payload["user_webhook_url"]
    
    # Verify task was enqueued
    mock_enqueue.assert_called_once()
    args, _ = mock_enqueue.call_args
    assert args[0] == data["deployment_id"]


@patch("src.api.routes.deployments.enqueue_orchestration_task", new_callable=AsyncMock)
def test_post_deployments_with_model_name_provider(
    mock_enqueue,
    client: TestClient,
    auth_headers: dict,
) -> None:
    """POST with model_name + provider (fal, veo3) resolves to HF ID and returns 202."""
    resp = client.post(
        "/v1/deployments",
        json={
            "model_name": "veo3",
            "provider": "fal",
            "user_runpod_key": "rpa_xxx",
            "user_webhook_url": "https://example.com/webhook",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["model_id"] == "black-forest-labs/FLUX.1-schnell"
    assert "deployment_id" in data
    
    mock_enqueue.assert_called_once()


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
    assert "deployment-orchestrator" in resp.json().get("service", "")
