"""Unit tests for config."""

import os

import pytest

from src.core.config import Settings, get_settings


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    s = Settings()
    assert s.gcp_project_id == "test-project"
    assert s.firestore_collection_deployments == "deployments"
    assert s.runpod_max_retries == 3


def test_settings_log_level_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "x")
    monkeypatch.setenv("LOG_LEVEL", "INVALID")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.log_level == "DEBUG"


def test_get_settings_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b
