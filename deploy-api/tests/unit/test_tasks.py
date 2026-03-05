"""Unit tests for Cloud Tasks orchestration and Secret Manager secret storage."""

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch, call

os.environ.setdefault("GCP_PROJECT_ID", "visgate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sm_client_mock():
    """Return a mock secretmanager.SecretManagerServiceClient."""
    client = MagicMock()
    client.create_secret.return_value = MagicMock(name="projects/visgate/secrets/visgate-dep-DEP1")
    client.add_secret_version.return_value = MagicMock()
    mock_response = MagicMock()
    mock_response.payload.data = json.dumps({"runpod_api_key": "rpa_test", "hf_token": None}).encode()
    client.access_secret_version.return_value = mock_response
    client.destroy_secret_version.return_value = MagicMock()
    return client


# ---------------------------------------------------------------------------
# _store_task_secrets
# ---------------------------------------------------------------------------

@patch("google.cloud.secretmanager.SecretManagerServiceClient")
def test_store_task_secrets_creates_secret_and_version(mock_sm_cls):
    """_store_task_secrets creates SM secret + version; returns secret name."""
    mock_client = _make_sm_client_mock()
    mock_sm_cls.return_value = mock_client

    from src.services.tasks import _store_task_secrets

    result = _store_task_secrets(
        "DEP1",
        "visgate",
        {"runpod_api_key": "rpa_key", "hf_token": "hf_tok", "aws_access_key_id": None,
         "aws_secret_access_key": None, "aws_endpoint_url": None, "s3_model_url": None},
    )

    assert result == "visgate-dep-DEP1"
    # create_secret must be called with the correct secret_id
    create_args = mock_client.create_secret.call_args[1]["request"]
    assert create_args["secret_id"] == "visgate-dep-DEP1"
    # add_secret_version must be called with the correct payload
    version_args = mock_client.add_secret_version.call_args[1]["request"]
    stored = json.loads(version_args["payload"]["data"].decode())
    assert stored["runpod_api_key"] == "rpa_key"
    assert stored["hf_token"] == "hf_tok"


@patch("google.cloud.secretmanager.SecretManagerServiceClient")
def test_store_task_secrets_continues_if_secret_exists(mock_sm_cls):
    """_store_task_secrets silently ignores AlreadyExists on create_secret."""
    mock_client = _make_sm_client_mock()
    mock_client.create_secret.side_effect = Exception("Already exists")
    mock_sm_cls.return_value = mock_client

    from src.services.tasks import _store_task_secrets

    # Should not raise
    result = _store_task_secrets("DEP2", "visgate", {"runpod_api_key": "k"})
    assert result == "visgate-dep-DEP2"
    # add_secret_version is still called when create fails
    mock_client.add_secret_version.assert_called_once()


# ---------------------------------------------------------------------------
# enqueue_orchestration_task — asyncio fallback
# ---------------------------------------------------------------------------

@patch("src.services.tasks.orchestrate_deployment", new_callable=AsyncMock)
@patch("asyncio.create_task")
async def test_enqueue_falls_back_to_asyncio_without_queue(mock_create_task, mock_orch, monkeypatch):
    """Without CLOUD_TASKS_QUEUE_PATH, enqueue_orchestration_task uses asyncio.create_task."""
    from src.core.config import get_settings
    import src.services.tasks as tasks_mod

    monkeypatch.delenv("CLOUD_TASKS_QUEUE_PATH", raising=False)
    get_settings.cache_clear()

    await tasks_mod.enqueue_orchestration_task("DEP_ASYNC", "rpa_key", None)

    mock_create_task.assert_called_once()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# enqueue_orchestration_task — Cloud Tasks path
# ---------------------------------------------------------------------------

@patch("src.services.tasks._store_task_secrets")
@patch("google.cloud.tasks_v2.CloudTasksClient")
@patch("src.services.tasks.orchestrate_deployment", new_callable=AsyncMock)
async def test_enqueue_cloud_tasks_hides_credentials(mock_orch, mock_tasks_cls, mock_store, monkeypatch):
    """
    When CLOUD_TASKS_QUEUE_PATH + INTERNAL_WEBHOOK_BASE_URL are set:
    - Secrets are sent to Secret Manager (not the task body)
    - Task body contains only deployment_id and secret_ref
    - Task body does NOT contain runpod_api_key, hf_token, or AWS keys
    """
    from src.core.config import get_settings
    import src.services.tasks as tasks_mod

    monkeypatch.setenv("CLOUD_TASKS_QUEUE_PATH", "projects/visgate/locations/us-central1/queues/q")
    monkeypatch.setenv("INTERNAL_WEBHOOK_BASE_URL", "https://service.run.app")
    get_settings.cache_clear()

    mock_store.return_value = "visgate-dep-DEP_CT"
    mock_client = MagicMock()
    mock_client.create_task.return_value = MagicMock(name="projects/visgate/.../tasks/t1")
    mock_tasks_cls.return_value = mock_client

    await tasks_mod.enqueue_orchestration_task(
        "DEP_CT", "rpa_secret_key", "hf_tok",
        "aws_key", "aws_secret", None, None,
    )

    # 1. Secrets stored in SM
    mock_store.assert_called_once()
    stored = mock_store.call_args[0][2]
    assert stored["runpod_api_key"] == "rpa_secret_key"
    assert stored["aws_access_key_id"] == "aws_key"

    # 2. Task body has no credentials
    task_arg = mock_client.create_task.call_args[1]["request"]["task"]
    body = json.loads(task_arg["http_request"]["body"])
    assert "runpod_api_key" not in body
    assert "hf_token" not in body
    assert "aws_access_key_id" not in body
    assert body["deployment_id"] == "DEP_CT"
    assert body["secret_ref"] == "visgate-dep-DEP_CT"

    # 3. URL targets the correct internal endpoint
    assert task_arg["http_request"]["url"] == "https://service.run.app/internal/tasks/orchestrate-deployment"

    get_settings.cache_clear()


@patch("src.services.tasks._store_task_secrets")
@patch("google.cloud.tasks_v2.CloudTasksClient")
@patch("src.services.tasks.orchestrate_deployment", new_callable=AsyncMock)
async def test_enqueue_cloud_tasks_adds_oidc_when_sa_configured(mock_orch, mock_tasks_cls, mock_store, monkeypatch):
    """OIDC token is attached to the task when CLOUD_TASKS_SERVICE_ACCOUNT is set."""
    from src.core.config import get_settings
    import src.services.tasks as tasks_mod

    monkeypatch.setenv("CLOUD_TASKS_QUEUE_PATH", "projects/visgate/locations/us-central1/queues/q")
    monkeypatch.setenv("INTERNAL_WEBHOOK_BASE_URL", "https://service.run.app")
    monkeypatch.setenv("CLOUD_TASKS_SERVICE_ACCOUNT", "tasks-sa@visgate.iam.gserviceaccount.com")
    get_settings.cache_clear()

    mock_store.return_value = "visgate-dep-DEP_OIDC"
    mock_client = MagicMock()
    mock_client.create_task.return_value = MagicMock(name="t1")
    mock_tasks_cls.return_value = mock_client

    await tasks_mod.enqueue_orchestration_task("DEP_OIDC", "rpa_k", None)

    task_arg = mock_client.create_task.call_args[1]["request"]["task"]
    assert task_arg["http_request"]["oidc_token"]["service_account_email"] == "tasks-sa@visgate.iam.gserviceaccount.com"
    assert task_arg["http_request"]["oidc_token"]["audience"] == "https://service.run.app"
    get_settings.cache_clear()


@patch("src.services.tasks._store_task_secrets")
@patch("google.cloud.tasks_v2.CloudTasksClient")
@patch("asyncio.create_task")
@patch("src.services.tasks.orchestrate_deployment", new_callable=AsyncMock)
async def test_enqueue_falls_back_to_asyncio_on_cloud_tasks_error(mock_orch, mock_create_task, mock_tasks_cls, mock_store, monkeypatch):
    """When Cloud Tasks raises, falls back gracefully to asyncio.create_task."""
    from src.core.config import get_settings
    import src.services.tasks as tasks_mod

    monkeypatch.setenv("CLOUD_TASKS_QUEUE_PATH", "projects/visgate/locations/us-central1/queues/q")
    monkeypatch.setenv("INTERNAL_WEBHOOK_BASE_URL", "https://service.run.app")
    get_settings.cache_clear()

    mock_store.return_value = "visgate-dep-DEP_FAIL"
    mock_client = MagicMock()
    mock_client.create_task.side_effect = Exception("GCP unavailable")
    mock_tasks_cls.return_value = mock_client

    await tasks_mod.enqueue_orchestration_task("DEP_FAIL", "rpa_k", None)

    # Should not raise; asyncio fallback used
    mock_create_task.assert_called_once()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# /internal/tasks/orchestrate-deployment endpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def internal_client(monkeypatch):
    """TestClient with mocked Firestore (same as main client fixture but lighter)."""
    from src.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("GCP_PROJECT_ID", "visgate")
    monkeypatch.setenv("INTERNAL_WEBHOOK_SECRET", "")

    # Inline the Firestore mock rather than importing from conftest
    _store: dict = {}
    mock_fs = MagicMock()
    def _collection(name):
        col = MagicMock()
        def _document(doc_id):
            ref = MagicMock()
            def _get():
                if doc_id not in _store:
                    r = MagicMock(); r.exists = False; return r
                r = MagicMock(); r.exists = True; r.to_dict.return_value = _store[doc_id]; return r
            ref.set = lambda data: _store.update({doc_id: data})
            ref.get = _get
            ref.update = lambda updates: _store[doc_id].update(updates) if doc_id in _store else None
            return ref
        col.document = _document
        return col
    mock_fs.collection = _collection

    from src.api import dependencies
    monkeypatch.setattr(dependencies, "get_firestore_client", lambda project_id=None: mock_fs)

    from src.main import app
    return TestClient(app)


@patch("src.api.routes.internal._fetch_and_destroy_task_secrets")
@patch("src.services.deployment.orchestrate_deployment", new_callable=AsyncMock)
@patch("asyncio.create_task")
def test_internal_orchestrate_endpoint_200(mock_create_task, mock_orch, mock_fetch, internal_client, monkeypatch):
    """POST /internal/tasks/orchestrate-deployment returns 200 and starts orchestration."""
    mock_fetch.return_value = {
        "runpod_api_key": "rpa_test",
        "hf_token": None,
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_endpoint_url": None,
        "s3_model_url": None,
    }

    resp = internal_client.post(
        "/internal/tasks/orchestrate-deployment",
        json={"deployment_id": "dep_2026_test", "secret_ref": "visgate-dep-dep_2026_test"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    assert resp.json()["deployment_id"] == "dep_2026_test"

    mock_fetch.assert_called_once_with("visgate-dep-dep_2026_test", "visgate")
    mock_create_task.assert_called_once()


@patch("src.api.routes.internal._fetch_and_destroy_task_secrets")
@pytest.mark.skip(reason="Secret header check disabled - OIDC auth is used instead")
def test_internal_orchestrate_endpoint_403_wrong_secret(mock_fetch, monkeypatch):
    """POST with wrong X-Visgate-Internal-Secret returns 403."""
    from src.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("INTERNAL_WEBHOOK_SECRET", "correct-secret")

    _store: dict = {}
    mock_fs = MagicMock()
    def _col(name):
        col = MagicMock()
        def _doc(doc_id):
            ref = MagicMock()
            ref.set = lambda data: None
            def _get():
                r = MagicMock(); r.exists = False; return r
            ref.get = _get
            ref.update = lambda u: None
            return ref
        col.document = _doc
        return col
    mock_fs.collection = _col

    from src.api import dependencies
    monkeypatch.setattr(dependencies, "get_firestore_client", lambda project_id=None: mock_fs)
    from src.main import app
    client = TestClient(app)

    resp = client.post(
        "/internal/tasks/orchestrate-deployment",
        json={"deployment_id": "dep_test", "secret_ref": "ref"},
        headers={"X-Visgate-Internal-Secret": "wrong-secret"},
    )
    assert resp.status_code == 403
    mock_fetch.assert_not_called()
    get_settings.cache_clear()


@patch("src.api.routes.internal._fetch_and_destroy_task_secrets")
def test_internal_orchestrate_endpoint_500_on_sm_error(mock_fetch, internal_client):
    """POST returns 500 when Secret Manager fetch fails."""
    mock_fetch.side_effect = Exception("SM unavailable")

    resp = internal_client.post(
        "/internal/tasks/orchestrate-deployment",
        json={"deployment_id": "dep_test", "secret_ref": "bad-ref"},
    )
    assert resp.status_code == 500
    assert "Failed to fetch task secrets" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _fetch_and_destroy_task_secrets
# ---------------------------------------------------------------------------

@patch("google.cloud.secretmanager.SecretManagerServiceClient")
def test_fetch_and_destroy_returns_secrets_and_destroys_version(mock_sm_cls):
    """_fetch_and_destroy_task_secrets returns parsed dict and calls destroy_secret_version."""
    mock_client = _make_sm_client_mock()
    mock_sm_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.payload.data = json.dumps({
        "runpod_api_key": "rpa_fetched",
        "hf_token": "hf_abc",
    }).encode()
    mock_client.access_secret_version.return_value = mock_response

    from src.api.routes.internal import _fetch_and_destroy_task_secrets
    result = _fetch_and_destroy_task_secrets("visgate-dep-DEP_F", "visgate")

    assert result["runpod_api_key"] == "rpa_fetched"
    assert result["hf_token"] == "hf_abc"
    mock_client.access_secret_version.assert_called_once()
    mock_client.destroy_secret_version.assert_called_once()


@patch("google.cloud.secretmanager.SecretManagerServiceClient")
def test_fetch_and_destroy_survives_destroy_failure(mock_sm_cls):
    """_fetch_and_destroy_task_secrets returns data even if destroy_secret_version fails."""
    mock_client = _make_sm_client_mock()
    mock_sm_cls.return_value = mock_client
    mock_client.destroy_secret_version.side_effect = Exception("Permission denied")

    from src.api.routes.internal import _fetch_and_destroy_task_secrets
    result = _fetch_and_destroy_task_secrets("visgate-dep-DEP_ND", "visgate")

    # Data is still returned despite destroy failure
    assert "runpod_api_key" in result
