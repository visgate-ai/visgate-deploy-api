"""Unit tests for GET /v1/deployments (list) and GET /v1/models."""

import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GCP_PROJECT_ID", "visgate")


# ---------------------------------------------------------------------------
# GET /v1/deployments (list)
# ---------------------------------------------------------------------------


def _make_doc(dep_id: str, api_key_id: str, status: str = "ready") -> dict:
    return {
        "deployment_id": dep_id,
        "api_key_id": api_key_id,
        "user_hash": "u_hash",
        "status": status,
        "hf_model_id": "black-forest-labs/FLUX.1-schnell",
        "endpoint_url": f"https://runpod.io/ep/{dep_id}",
        "gpu_allocated": "NVIDIA A40",
        "model_vram_gb": 16,
        "runpod_endpoint_id": f"ep_{dep_id}",
        "logs": [],
        "error": None,
        "created_at": datetime.now(UTC).isoformat(),
        "ready_at": None,
        "phase_timestamps": {},
    }


def test_list_deployments_empty(client: TestClient, auth_headers: dict) -> None:
    """GET /v1/deployments returns 200 with empty list when no docs exist."""
    with patch("src.api.routes.deployments.list_deployments") as mock_list:
        mock_list.return_value = []
        resp = client.get("/v1/deployments", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["deployments"] == []
    assert data["total"] == 0
    assert data["limit"] == 20


def test_list_deployments_returns_items(client: TestClient, auth_headers: dict) -> None:
    """GET /v1/deployments returns correct items for authenticated API key."""
    from src.models.entities import DeploymentDoc

    doc = DeploymentDoc.from_firestore_dict(_make_doc("dep_2024_abc1", "key_test1"))

    with patch("src.api.routes.deployments.list_deployments") as mock_list:
        mock_list.return_value = [doc]
        resp = client.get("/v1/deployments", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["deployments"][0]["deployment_id"] == "dep_2024_abc1"
    assert data["deployments"][0]["status"] == "ready"


def test_list_deployments_status_filter_forwarded(
    client: TestClient, auth_headers: dict
) -> None:
    """status query param is forwarded to list_deployments."""
    with patch("src.api.routes.deployments.list_deployments") as mock_list:
        mock_list.return_value = []
        resp = client.get(
            "/v1/deployments?deployment_status=ready", headers=auth_headers
        )

    assert resp.status_code == 200
    _, kwargs = mock_list.call_args
    assert kwargs.get("status_filter") == "ready" or mock_list.call_args[0][3] == "ready"


def test_list_deployments_limit_capped(client: TestClient, auth_headers: dict) -> None:
    """limit is capped at 100 even if client sends a larger value."""
    with patch("src.api.routes.deployments.list_deployments") as mock_list:
        mock_list.return_value = []
        resp = client.get("/v1/deployments?limit=9999", headers=auth_headers)

    assert resp.status_code == 200
    # The mock was called; check the limit passed in was <= 100
    called_limit = mock_list.call_args[1].get("limit") or mock_list.call_args[0][4]
    assert called_limit <= 100


def test_list_deployments_401_without_auth(client: TestClient) -> None:
    """GET /v1/deployments without auth returns 401."""
    resp = client.get("/v1/deployments")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


def test_list_models_no_r2(client: TestClient) -> None:
    """GET /v1/models returns all registry models with cached=False when R2 off."""
    with patch("src.api.routes.models.fetch_cached_model_ids") as mock_r2:
        mock_r2.return_value = set()
        resp = client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    assert isinstance(data["models"], list)

    model_ids = {m["model_id"] for m in data["models"]}
    assert "black-forest-labs/FLUX.1-schnell" in model_ids

    # All cached=False when R2 not configured
    assert all(not m["cached"] for m in data["models"])


def test_list_models_with_r2_cache(client: TestClient, monkeypatch) -> None:
    """GET /v1/models marks cached=True for models in R2 manifest."""
    monkeypatch.setenv("VISGATE_DEPLOYAPI_R2_ACCESS_KEY_ID_RO", "rtest")
    monkeypatch.setenv("VISGATE_DEPLOYAPI_R2_SECRET_ACCESS_KEY_RO", "rsecret")
    monkeypatch.setenv("VISGATE_DEPLOYAPI_R2_ENDPOINT_URL", "https://r2.example.com")

    cached_id = "black-forest-labs/FLUX.1-schnell"

    with patch("src.api.routes.models.fetch_cached_model_ids") as mock_r2:
        mock_r2.return_value = {cached_id}
        # Re-import after env change so settings pick up new values
        from src.core.config import get_settings
        get_settings.cache_clear()
        resp = client.get("/v1/models")
        get_settings.cache_clear()  # clean up

    assert resp.status_code == 200
    data = resp.json()
    cached_entries = [m for m in data["models"] if m["cached"]]
    assert any(m["model_id"] == cached_id for m in cached_entries)


def test_list_models_has_required_fields(client: TestClient) -> None:
    """Each model entry has model_id, tasks, gpu_memory_gb, cached."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    for entry in resp.json()["models"]:
        assert "model_id" in entry
        assert "tasks" in entry
        assert "gpu_memory_gb" in entry
        assert "cached" in entry


def test_list_models_no_auth_required(client: TestClient) -> None:
    """GET /v1/models is public — no Authorization header needed."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200


def test_list_models_cached_entries_sorted_first(client: TestClient) -> None:
    """Cached models appear before non-cached models in the response."""
    cached_id = "black-forest-labs/FLUX.1-schnell"
    with patch("src.api.routes.models.fetch_cached_model_ids") as mock_r2:
        mock_r2.return_value = {cached_id}

        from src.core.config import get_settings
        import src.core.config as cfg_mod
        fake_settings = MagicMock()
        fake_settings.r2_endpoint_url = "https://r2.example.com"
        fake_settings.r2_access_key_id_ro = "rtest"
        fake_settings.r2_secret_access_key_ro = "rsecret"
        with patch.object(cfg_mod, "get_settings", return_value=fake_settings):
            resp = client.get("/v1/models")

    assert resp.status_code == 200
    models = resp.json()["models"]
    if len(models) > 1:
        cached_indices = [i for i, m in enumerate(models) if m["cached"]]
        uncached_indices = [i for i, m in enumerate(models) if not m["cached"]]
        if cached_indices and uncached_indices:
            assert max(cached_indices) < min(uncached_indices)
