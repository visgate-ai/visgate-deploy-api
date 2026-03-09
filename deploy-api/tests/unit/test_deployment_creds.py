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
os.environ.setdefault("RUNPOD_TEMPLATE_ID", "tpl-default")


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
    doc.task = None
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
    stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, return_value={"id": "tpl-default"}))
    stack.enter_context(patch("src.services.deployment._probe_runpod_readiness", new_callable=AsyncMock, return_value=(True, None)))
    stack.enter_context(patch("src.services.deployment.get_provider", return_value=provider_mock))


def _make_template_capturer():
    """Returns (captured_env dict, async mock fn) that converts template env list → dict.

    RunPod env is passed to create_serverless_template as a list of
    {"key": k, "value": v} dicts.  This helper converts that into a plain
    Python dict so tests can assert on it easily.
    """
    captured: dict = {}

    async def _capturer(**kwargs):
        for item in kwargs.get("env", []):
            captured[item["key"]] = item["value"]
        return {"id": "dep-tpl-test"}

    return captured, _capturer


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

    # Env is now baked into the RunPod template, not the endpoint; capture from there.
    captured_env, template_capturer = _make_template_capturer()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep1", "url": "https://api.runpod.ai/v2/ep1/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, side_effect=template_capturer))
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_test",
            runpod_api_key="rpa_userkey",
            aws_access_key_id=None,
            aws_secret_access_key=None,
        )

    assert captured_env.get("VISGATE_R2_ACCESS_KEY_ID") == "ro_key_EXPECTED"
    assert captured_env.get("VISGATE_R2_SECRET_ACCESS_KEY") == "ro_secret_EXPECTED"
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

    captured_env, template_capturer = _make_template_capturer()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep2", "url": "https://api.runpod.ai/v2/ep2/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, side_effect=template_capturer))
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_private",
            runpod_api_key="rpa_userkey",
            aws_access_key_id="user_private_key",
            aws_secret_access_key="user_private_secret",
        )

    assert captured_env.get("VISGATE_R2_ACCESS_KEY_ID") == "user_private_key"
    assert captured_env.get("VISGATE_R2_SECRET_ACCESS_KEY") == "user_private_secret"
    assert "ro_platform_key" not in captured_env.values()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_runpod_api_key_in_worker_env(monkeypatch):
    """RUNPOD_API_KEY is always injected into worker env for self-cleanup."""
    from src.core.config import get_settings
    get_settings.cache_clear()

    captured_env, template_capturer = _make_template_capturer()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep3", "url": "https://api.runpod.ai/v2/ep3/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, side_effect=template_capturer))
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

    captured_env, template_capturer = _make_template_capturer()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep3", "url": "https://api.runpod.ai/v2/ep3/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, side_effect=template_capturer))
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

    captured_env, template_capturer = _make_template_capturer()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep4", "url": "https://api.runpod.ai/v2/ep4/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, side_effect=template_capturer))
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

    captured_env, template_capturer = _make_template_capturer()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep5", "url": "https://api.runpod.ai/v2/ep5/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, side_effect=template_capturer))
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_hf_override",
            runpod_api_key="rpa_key",
            hf_token_override="hf_caller_token",
        )

    assert captured_env.get("HF_TOKEN") == "hf_caller_token"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_worker_target_is_persisted_and_logged(monkeypatch):
    """Selected worker profile/template/image should be visible in deployment state and logs."""
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID", "tpl-default")
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID_AUDIO", "tpl-audio")
    monkeypatch.setenv("DOCKER_IMAGE", "visgateai/inference:latest")
    monkeypatch.setenv("DOCKER_IMAGE_AUDIO", "visgateai/inference-audio:latest")
    get_settings.cache_clear()

    update_mock = MagicMock()
    append_log_mock = MagicMock()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep-audio", "url": "https://api.runpod.ai/v2/ep-audio/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                return_value=_make_deployment_doc(hf_model_id="openai/whisper-large-v3"),
            )
        )
        stack.enter_context(patch("src.services.deployment.update_deployment", update_mock))
        stack.enter_context(patch("src.services.deployment.get_gpu_registry", return_value=[]))
        stack.enter_context(patch("src.services.deployment.get_tier_mapping", return_value={}))
        stack.enter_context(patch("src.services.deployment.select_gpu_candidates", return_value=[("GPU_A40", "A40")]))
        stack.enter_context(patch("src.services.deployment.get_firestore_client", return_value=MagicMock()))
        stack.enter_context(patch("src.services.deployment.validate_model", new_callable=AsyncMock, return_value=_make_model_info(10)))
        stack.enter_context(patch("src.services.deployment.append_log", append_log_mock))
        stack.enter_context(patch("src.services.deployment.notify", new_callable=AsyncMock, return_value=True))
        stack.enter_context(patch("src.services.deployment.get_secrets", return_value=None))
        stack.enter_context(patch("src.services.deployment._probe_runpod_readiness", new_callable=AsyncMock, return_value=(True, None)))
        stack.enter_context(patch("src.services.deployment.get_provider", return_value=mock_provider))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_audio",
            runpod_api_key="rpa_audio",
        )

    assert any(
        call.args[3].get("worker_profile") == "audio"
        and call.args[3].get("worker_template_id") == "tpl-audio"
        and call.args[3].get("worker_image") == "visgateai/inference-audio:latest"
        for call in update_mock.call_args_list
    )
    assert any(
        "worker_profile=audio" in call.args[4]
        and "image=visgateai/inference-audio:latest" in call.args[4]
        for call in append_log_mock.call_args_list
    )
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_video_deployment_uses_warm_worker_and_extended_load_wait(monkeypatch):
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID_VIDEO", "tpl-video")
    monkeypatch.setenv("DOCKER_IMAGE_VIDEO", "visgateai/inference-video:latest")
    monkeypatch.setenv("RUNPOD_WORKERS_MIN", "0")
    monkeypatch.setenv("RUNPOD_WORKERS_MIN_VIDEO", "1")
    monkeypatch.setenv("RUNPOD_WORKERS_MAX", "1")
    monkeypatch.setenv("RUNPOD_EXECUTION_TIMEOUT_MS_VIDEO", "900000")
    get_settings.cache_clear()

    captured_kwargs: dict = {}
    sync_template_mock = AsyncMock(return_value={"id": "tpl-video"})

    async def mock_create_endpoint(**kwargs):
        captured_kwargs.update(kwargs)
        return {"id": "ep-video", "url": "https://api.runpod.ai/v2/ep-video/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)
    initial_doc = _make_deployment_doc(hf_model_id="Wan-AI/Wan2.1-T2V-1.3B")
    ready_doc = _make_deployment_doc(
        hf_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        status="ready",
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                side_effect=[initial_doc, ready_doc],
            )
        )
        stack.enter_context(patch("src.services.deployment.update_deployment"))
        stack.enter_context(patch("src.services.deployment.get_gpu_registry", return_value=[]))
        stack.enter_context(patch("src.services.deployment.get_tier_mapping", return_value={}))
        stack.enter_context(patch("src.services.deployment.select_gpu_candidates", return_value=[("GPU_A40", "A40")]))
        stack.enter_context(patch("src.services.deployment.get_firestore_client", return_value=MagicMock()))
        stack.enter_context(patch("src.services.deployment.validate_model", new_callable=AsyncMock, return_value=_make_model_info(24)))
        stack.enter_context(patch("src.services.deployment.append_log"))
        stack.enter_context(patch("src.services.deployment.notify", new_callable=AsyncMock, return_value=True))
        stack.enter_context(patch("src.services.deployment.get_secrets", return_value=None))
        stack.enter_context(patch("src.services.deployment.create_serverless_template", sync_template_mock))
        stack.enter_context(patch("src.services.deployment._probe_runpod_readiness", new_callable=AsyncMock, return_value=(True, None)))
        stack.enter_context(patch("src.services.deployment.get_provider", return_value=mock_provider))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_video",
            runpod_api_key="rpa_video",
        )

    assert captured_kwargs["workers_min"] == 1
    assert captured_kwargs["execution_timeout_ms"] == 900000
    # Env is now baked into the per-deployment RunPod template, not the endpoint.
    assert captured_kwargs["env"] == {}
    assert sync_template_mock.await_count == 1
    tpl_call_kwargs = sync_template_mock.call_args.kwargs
    assert tpl_call_kwargs["api_key"] == "rpa_video"
    assert tpl_call_kwargs["image_name"] == "visgateai/inference-video:latest"
    assert tpl_call_kwargs["container_disk_in_gb"] == 50
    tpl_env = {e["key"]: e["value"] for e in tpl_call_kwargs["env"]}
    assert tpl_env["HF_MODEL_ID"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert tpl_env["MODEL_LOAD_WAIT_TIMEOUT_SECONDS"] == "840"
    assert tpl_env["RUNPOD_INIT_TIMEOUT"] == "900"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_video_deployment_does_not_use_health_probe_readiness(monkeypatch):
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID_VIDEO", "tpl-video")
    monkeypatch.setenv("DOCKER_IMAGE_VIDEO", "visgateai/inference-video:latest")
    get_settings.cache_clear()

    initial_doc = _make_deployment_doc(hf_model_id="Wan-AI/Wan2.1-T2V-1.3B")
    ready_doc = _make_deployment_doc(
        hf_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        status="ready",
    )
    probe_mock = AsyncMock(return_value=(True, None))

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep-video", "url": "https://api.runpod.ai/v2/ep-video/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                side_effect=[initial_doc, ready_doc],
            )
        )
        stack.enter_context(patch("src.services.deployment.update_deployment"))
        stack.enter_context(patch("src.services.deployment.get_gpu_registry", return_value=[]))
        stack.enter_context(patch("src.services.deployment.get_tier_mapping", return_value={}))
        stack.enter_context(patch("src.services.deployment.select_gpu_candidates", return_value=[("GPU_A40", "A40")]))
        stack.enter_context(patch("src.services.deployment.get_firestore_client", return_value=MagicMock()))
        stack.enter_context(patch("src.services.deployment.validate_model", new_callable=AsyncMock, return_value=_make_model_info(24)))
        stack.enter_context(patch("src.services.deployment.append_log"))
        stack.enter_context(patch("src.services.deployment.notify", new_callable=AsyncMock, return_value=True))
        stack.enter_context(patch("src.services.deployment.get_secrets", return_value=None))
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, return_value={"id": "tpl-video"}))
        stack.enter_context(patch("src.services.deployment._probe_runpod_readiness", probe_mock))
        stack.enter_context(patch("src.services.deployment.get_provider", return_value=mock_provider))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_video",
            runpod_api_key="rpa_video",
        )

    probe_mock.assert_not_awaited()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_video_deployment_uses_r2_cache_hit_s3_url(monkeypatch):
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID_VIDEO", "tpl-video")
    monkeypatch.setenv("DOCKER_IMAGE_VIDEO", "visgateai/inference-video:latest")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_RW", "rw_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_RW", "rw_secret")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_ACCESS_KEY_ID_R", "ro_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_R2_SECRET_ACCESS_KEY_R", "ro_secret")
    monkeypatch.setenv("VISGATE_DEPLOY_API_S3_API_R2", "https://acct.r2.cloudflarestorage.com")
    get_settings.cache_clear()

    captured_kwargs: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_kwargs.update(kwargs)
        return {"id": "ep-video", "url": "https://api.runpod.ai/v2/ep-video/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)
    initial_doc = _make_deployment_doc(hf_model_id="Wan-AI/Wan2.1-T2V-1.3B")
    ready_doc = _make_deployment_doc(
        hf_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        status="ready",
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                side_effect=[initial_doc, ready_doc],
            )
        )
        stack.enter_context(patch("src.services.deployment.update_deployment"))
        stack.enter_context(patch("src.services.deployment.get_gpu_registry", return_value=[]))
        stack.enter_context(patch("src.services.deployment.get_tier_mapping", return_value={}))
        stack.enter_context(patch("src.services.deployment.select_gpu_candidates", return_value=[("GPU_A40", "A40")]))
        stack.enter_context(patch("src.services.deployment.get_firestore_client", return_value=MagicMock()))
        stack.enter_context(patch("src.services.deployment.validate_model", new_callable=AsyncMock, return_value=_make_model_info(24)))
        stack.enter_context(patch("src.services.deployment.append_log"))
        stack.enter_context(patch("src.services.deployment.notify", new_callable=AsyncMock, return_value=True))
        stack.enter_context(patch("src.services.deployment.get_secrets", return_value=None))
        stack.enter_context(patch("src.services.deployment.fetch_cached_model_ids", return_value=["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"]))
        sync_template_mock_r2 = AsyncMock(return_value={"id": "tpl-video"})
        stack.enter_context(patch("src.services.deployment.create_serverless_template", sync_template_mock_r2))
        stack.enter_context(patch("src.services.deployment._probe_runpod_readiness", new_callable=AsyncMock, return_value=(True, None)))
        stack.enter_context(patch("src.services.deployment.get_provider", return_value=mock_provider))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_video",
            runpod_api_key="rpa_video",
        )

    # Env is now baked into the per-deployment RunPod template, not the endpoint.
    assert captured_kwargs["env"] == {}
    tpl_env = {e["key"]: e["value"] for e in sync_template_mock_r2.call_args.kwargs["env"]}
    # R2 cache is now enabled for video; on a cache HIT the worker gets S3_MODEL_URL
    assert "S3_MODEL_URL" in tpl_env
    assert tpl_env["HF_MODEL_ID"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    get_settings.cache_clear()

