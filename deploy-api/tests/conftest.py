"""Pytest configuration and shared fixtures."""

import os

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Ensure src is on path and GCP env for tests
os.environ.setdefault("GCP_PROJECT_ID", "visgate")
os.environ.setdefault("LOG_FORMAT", "readable")

# In-memory store for mocked Firestore
_firestore_store: dict[str, dict] = {}


def _make_firestore_mock() -> MagicMock:
    """Return a mock Firestore client that stores docs in _firestore_store."""
    client = MagicMock()
    def collection(name: str):
        col = MagicMock()
        def document(doc_id: str):
            doc_ref = MagicMock()
            def set(data: dict) -> None:
                _firestore_store[doc_id] = data
            def get() -> MagicMock:
                if doc_id not in _firestore_store:
                    result = MagicMock()
                    result.exists = False
                    return result
                result = MagicMock()
                result.exists = True
                result.to_dict.return_value = _firestore_store[doc_id]
                return result
            def update(updates: dict) -> None:
                if doc_id in _firestore_store:
                    _firestore_store[doc_id].update(updates)
            doc_ref.set = set
            doc_ref.get = get
            doc_ref.update = update
            return doc_ref
        col.document = document
        return col
    client.collection = collection
    return client


@pytest.fixture
def firestore_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject mock Firestore into app."""
    _firestore_store.clear()
    mock_client = _make_firestore_mock()
    from src.api import dependencies
    monkeypatch.setattr(dependencies, "get_firestore_client", lambda project_id=None: mock_client)
    return mock_client


@pytest.fixture
def client(firestore_mock: MagicMock) -> TestClient:
    """FastAPI test client with mocked Firestore."""
    from src.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict:
    """Bearer token for tests (Runpod API key)."""
    return {"Authorization": "Bearer rpa_test_key_placeholder"}


@pytest.fixture
def deployment_create_payload() -> dict:
    """Minimal valid POST /v1/deployments body."""
    return {
        "hf_model_id": "black-forest-labs/FLUX.1-schnell",
        "user_runpod_key": "rpa_test_key_placeholder",
        "user_webhook_url": "https://example.com/webhook",
        "gpu_tier": "A40",
    }
