"""Unit tests for config."""


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


# ── New VISGATE_DEPLOY_API_ env var aliases ─────────────────────────────────

def test_r2_rw_key_reads_from_visgate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """r2_access_key_id_rw and r2_secret_access_key_rw read via VISGATE_DEPLOY_API_ prefix."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_RW", "rw_key_id")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_RW", "rw_secret")
    s = Settings()
    assert s.r2_access_key_id_rw == "rw_key_id"
    assert s.r2_secret_access_key_rw == "rw_secret"


def test_r2_rw_key_falls_back_to_old_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old AWS_ACCESS_KEY_ID env var still works when VISGATE_ alias is unset."""
    monkeypatch.delenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_RW", raising=False)
    monkeypatch.delenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_RW", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "old_key_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "old_secret")
    s = Settings(_env_file=None)  # skip .env file to avoid stale values
    assert s.r2_access_key_id_rw == "old_key_id"
    assert s.r2_secret_access_key_rw == "old_secret"


def test_r2_ro_key_reads_from_visgate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """r2_access_key_id_ro / r2_secret_access_key_ro read via VISGATE_DEPLOY_API_ prefix."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_R", "ro_key_id")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_R", "ro_secret")
    s = Settings()
    assert s.r2_access_key_id_ro == "ro_key_id"
    assert s.r2_secret_access_key_ro == "ro_secret"


def test_hf_pro_token_reads_from_visgate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """hf_pro_access_token reads via VISGATE_DEPLOY_API_ prefix."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_HF_PRO_ACCESS_TOKEN", "hf_platform_token")
    s = Settings()
    assert s.hf_pro_access_token == "hf_platform_token"


def test_s3_model_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """r2_model_base_url defaults to s3://visgate-models/models."""
    monkeypatch.delenv("VISGATE_DEPLOY_API_S3_MODEL_URL_R2", raising=False)
    monkeypatch.delenv("S3_MODEL_URL", raising=False)
    s = Settings()
    assert s.r2_model_base_url == "s3://visgate-models/models"


def test_s3_model_url_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """r2_model_base_url can be overridden via VISGATE_DEPLOY_API_S3_MODEL_URL_R2."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_S3_MODEL_URL_R2", "s3://my-bucket/models")
    s = Settings()
    assert s.r2_model_base_url == "s3://my-bucket/models"


def test_aws_endpoint_reads_from_visgate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """r2_endpoint_url reads via VISGATE_DEPLOY_API_S3_API_R2."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_S3_API_R2", "https://acct.r2.cloudflarestorage.com")
    s = Settings()
    assert s.r2_endpoint_url == "https://acct.r2.cloudflarestorage.com"
