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
    """r2_access_key_id_rw and r2_secret_access_key_rw read from output RW canonical names."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", "rw_key_id")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", "rw_secret")
    s = Settings()
    assert s.r2_access_key_id_rw == "rw_key_id"
    assert s.r2_secret_access_key_rw == "rw_secret"


def test_r2_rw_key_reads_from_output_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", "rw_out_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", "rw_out_secret")
    s = Settings(_env_file=None)
    assert s.r2_access_key_id_rw == "rw_out_key"
    assert s.r2_secret_access_key_rw == "rw_out_secret"


def test_r2_rw_key_no_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy AWS fallback is disabled for security reasons."""
    monkeypatch.delenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", raising=False)
    monkeypatch.delenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", raising=False)
    s = Settings(_env_file=None)
    assert s.r2_access_key_id_rw == ""
    assert s.r2_secret_access_key_rw == ""


def test_r2_ro_key_reads_from_visgate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """r2_access_key_id_ro / r2_secret_access_key_ro read from input R canonical names."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_INPUT_R", "ro_key_id")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_INPUT_R", "ro_secret")
    s = Settings()
    assert s.r2_access_key_id_ro == "ro_key_id"
    assert s.r2_secret_access_key_ro == "ro_secret"


def test_r2_ro_key_reads_from_input_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_INPUT_R", "ro_in_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_INPUT_R", "ro_in_secret")
    s = Settings(_env_file=None)
    assert s.r2_access_key_id_ro == "ro_in_key"
    assert s.r2_secret_access_key_ro == "ro_in_secret"


def test_smoke_test_keys_and_bucket_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISGATE_DEPLOY_API_SMOKE_TEST_RUNPOD", "rpa_test")
    monkeypatch.setenv("VISGATE_DEPLOY_API_SMOKE_TEST_HF_KEY", "hf_test")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_BUCKET_NAME_OUTPUT", "bucket-output")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_BUCKET_NAME_INPUT", "bucket-input")
    s = Settings(_env_file=None)
    assert s.smoke_test_runpod_api_key == "rpa_test"
    assert s.smoke_test_hf_key == "hf_test"
    assert s.inference_r2_bucket_name_output == "bucket-output"
    assert s.inference_r2_bucket_name_input == "bucket-input"


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
    """r2_endpoint_url reads default endpoint when no env override is used."""
    monkeypatch.delenv("VISGATE_DEPLOY_API_R2_ENDPOINT_URL", raising=False)
    s = Settings()
    assert s.r2_endpoint_url == "https://088e0d2618d33e55a76e4d65b439d6c4.r2.cloudflarestorage.com"


def test_legacy_secret_env_names_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_RW", "legacy")
    with pytest.raises(ValueError, match="Legacy secret/env names are not allowed"):
        s = Settings(_env_file=None)
        s.resolve_secrets()


def test_root_path_reads_from_visgate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """root_path can be set for public path-prefix publishing."""
    monkeypatch.setenv("VISGATE_DEPLOY_API_ROOT_PATH", "/deployapi")
    s = Settings()
    assert s.root_path == "/deployapi"


def test_modality_template_aliases_read_from_visgate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISGATE_DEPLOY_API_RUNPOD_TEMPLATE_ID_IMAGE", "tpl-image")
    monkeypatch.setenv("VISGATE_DEPLOY_API_RUNPOD_TEMPLATE_ID_AUDIO", "tpl-audio")
    monkeypatch.setenv("VISGATE_DEPLOY_API_RUNPOD_TEMPLATE_ID_VIDEO", "tpl-video")
    s = Settings()
    assert s.runpod_template_id_image == "tpl-image"
    assert s.runpod_template_id_audio == "tpl-audio"
    assert s.runpod_template_id_video == "tpl-video"


def test_shared_cache_allowlist_includes_validated_audio_and_video_models() -> None:
    s = Settings()
    assert "openai/whisper-large-v3" in s.shared_cache_allowed_models
    assert "Wan-AI/Wan2.1-T2V-1.3B" in s.shared_cache_allowed_models
    assert "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" in s.shared_cache_allowed_models


def test_video_workers_min_defaults_to_one() -> None:
    s = Settings()
    assert s.runpod_workers_min == 1
    assert s.runpod_workers_min_video == 1
