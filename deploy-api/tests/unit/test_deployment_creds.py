"""Tests for credential injection in orchestrate_deployment.

Covers:
- R2 RO key sent to workers (RW key is NOT sent)
- HF platform token auto-injected when caller omits hf_token
- RUNPOD_API_KEY env var present in worker env for self-cleanup
- User private keys override platform R2 keys
"""
from __future__ import annotations

import os
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GCP_PROJECT_ID", "visgate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deployment_doc(
    hf_model_id: str = "stabilityai/sd-turbo",
    status: str = "validating",
    internal_webhook_base_url: str | None = None,
):
    doc = MagicMock()
    doc.hf_model_id = hf_model_id
    doc.status = status
    doc.gpu_tier = None
    doc.endpoint_name = None
    doc.region = None
    doc.runpod_endpoint_id = "ep_test"
    doc.endpoint_url = "https://api.runpod.ai/v2/ep_test/run"
    doc.internal_webhook_base_url = internal_webhook_base_url
    return doc


def _make_model_info(vram_gb: int = 8):
    info = MagicMock()
    info.min_gpu_memory_gb = vram_gb
    return info


def _apply_base_patches(stack: ExitStack, provider_mock):
    """Enter all common patches through an ExitStack."""
    stack.enter_context(patch("src.services.deployment.get_deployment", return_value=_make_deployment_doc()))
    stack.enter_context(patch("src.services.deployment.update_deployment"))
    stack.enter_context(patch("src.services.deployment.get_gpu_registry", return_value=[]))
    stack.enter_context(patch("src.services.deployment.get_tier_mapping", return_value={}))
    stack.enter_context(patch("src.services.deployment.select_gpu_candidates", return_value=[("GPU_A40", "A40")]))
    stack.enter_context(patch("src.services.deployment.get_firestore_client", return_value=MagicMock()))
    stack.enter_context(patch("src.services.deployment.validate_model", new_callable=AsyncMock, return_value=_make_model_info()))
    stack.enter_context(patch("src.services.deployment.append_log"))
    stack.enter_context(patch("src.services.deployment.notify", new_callable=AsyncMock, return_value=True))
    stack.enter_context(patch("src.services.deployment.get_secrets", return_value=None))
    stack.enter_context(patch("src.services.deployment._probe_runpod_readiness", new_callable=AsyncMock, return_value=(True, None)))
    stack.enter_context(patch("src.services.deployment.get_provider", return_value=provider_mock))


# ---------------------------------------------------------------------------
# R2 key injection: RO key → worker, RW key stays in API
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_gets_r2_ro_key_not_rw(monkeypatch):
    """When shared cache is used, worker env gets RO key, never the RW API key."""
    from src.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_RW", "rw_key_SHOULD_NOT_BE_IN_WORKER")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_RW", "rw_secret_SHOULD_NOT_BE_IN_WORKER")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_R", "ro_key_EXPECTED")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_R", "ro_secret_EXPECTED")
    monkeypatch.setenv("VISGATE_DEPLOY_API_S3_API_R2", "https://acct.r2.cloudflarestorage.com")
    get_settings.cache_clear()

    captured_env: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_env.update(kwargs.get("env", {}))
        return {"id": "ep1", "url": "https://api.runpod.ai/v2/ep1/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_test",
            runpod_api_key="rpa_userkey",
            aws_access_key_id=None,
            aws_secret_access_key=None,
        )

    assert captured_env.get("AWS_ACCESS_KEY_ID") == "ro_key_EXPECTED"
    assert captured_env.get("AWS_SECRET_ACCESS_KEY") == "ro_secret_EXPECTED"
    assert "rw_key_SHOULD_NOT_BE_IN_WORKER" not in captured_env.values()
    assert "rw_secret_SHOULD_NOT_BE_IN_WORKER" not in captured_env.values()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_worker_gets_user_private_key_over_platform_ro(monkeypatch):
    """User's private keys take precedence over platform R2 RO key."""
    from src.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_R", "ro_platform_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_R", "ro_platform_secret")
    get_settings.cache_clear()

    captured_env: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_env.update(kwargs.get("env", {}))
        return {"id": "ep2", "url": "https://api.runpod.ai/v2/ep2/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_private",
            runpod_api_key="rpa_userkey",
            aws_access_key_id="user_private_key",
            aws_secret_access_key="user_private_secret",
        )

    assert captured_env.get("AWS_ACCESS_KEY_ID") == "user_private_key"
    assert captured_env.get("AWS_SECRET_ACCESS_KEY") == "user_private_secret"
    assert "ro_platform_key" not in captured_env.values()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_runpod_api_key_in_worker_env(monkeypatch):
    """RUNPOD_API_KEY is always injected into worker env for self-cleanup."""
    from src.core.config import get_settings
    get_settings.cache_clear()

    captured_env: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_env.update(kwargs.get("env", {}))
        return {"id": "ep3", "url": "https://api.runpod.ai/v2/ep3/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_cleanup",
            runpod_api_key="rpa_cleanupkey",
        )

    assert captured_env.get("RUNPOD_API_KEY") == "rpa_cleanupkey"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_worker_gets_absolute_internal_callback_urls_from_doc(monkeypatch):
    """Deployment docs can provide absolute callback URLs when env base URL is absent."""
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("INTERNAL_WEBHOOK_BASE_URL", raising=False)
    monkeypatch.setenv("INTERNAL_WEBHOOK_SECRET", "internal-secret")
    get_settings.cache_clear()

    captured_env: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_env.update(kwargs.get("env", {}))
        return {"id": "ep3", "url": "https://api.runpod.ai/v2/ep3/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                return_value=_make_deployment_doc(internal_webhook_base_url="https://api.example.com"),
            )
        )
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_cleanup",
            runpod_api_key="rpa_cleanupkey",
        )

    assert captured_env.get("VISGATE_WEBHOOK") == "https://api.example.com/internal/deployment-ready/dep_cleanup"
    assert captured_env.get("VISGATE_LOG_TUNNEL") == "https://api.example.com/internal/logs/dep_cleanup"
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# HF platform token fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hf_platform_token_injected_when_caller_omits_token(monkeypatch):
    """Platform HF token is used when caller provides no hf_token."""
    from src.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("VISGATE_DEPLOY_API_HF_PRO_ACCESS_TOKEN", "hf_platform_xyz")
    get_settings.cache_clear()

    captured_env: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_env.update(kwargs.get("env", {}))
        return {"id": "ep4", "url": "https://api.runpod.ai/v2/ep4/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_hf",
            runpod_api_key="rpa_key",
            hf_token_override=None,
        )

    assert captured_env.get("HF_TOKEN") == "hf_platform_xyz"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_caller_hf_token_overrides_platform_token(monkeypatch):
    """Caller-provided hf_token takes precedence over platform token."""
    from src.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("VISGATE_DEPLOY_API_HF_PRO_ACCESS_TOKEN", "hf_platform_xyz")
    get_settings.cache_clear()

    captured_env: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_env.update(kwargs.get("env", {}))
        return {"id": "ep5", "url": "https://api.runpod.ai/v2/ep5/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_hf_override",
            runpod_api_key="rpa_key",
            hf_token_override="hf_caller_token",
        )

    assert captured_env.get("HF_TOKEN") == "hf_caller_token"
    get_settings.cache_clear()

