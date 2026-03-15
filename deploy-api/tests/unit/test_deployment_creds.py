"""Tests for credential injection in orchestrate_deployment.

Covers:
- R2 RO key sent to workers (RW key is NOT sent)
- caller HF token requirement and injection
- RUNPOD_API_KEY env var present in worker env for self-cleanup
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
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", "rw_key_SHOULD_NOT_BE_IN_WORKER")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", "rw_secret_SHOULD_NOT_BE_IN_WORKER")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_INPUT_R", "ro_key_EXPECTED")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_INPUT_R", "ro_secret_EXPECTED")
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
            hf_token_override="hf_user_token",
        )

    assert captured_env.get("VISGATE_R2_ACCESS_KEY_ID") == "ro_key_EXPECTED"
    assert captured_env.get("VISGATE_R2_SECRET_ACCESS_KEY") == "ro_secret_EXPECTED"
    assert "rw_key_SHOULD_NOT_BE_IN_WORKER" not in captured_env.values()
    assert "rw_secret_SHOULD_NOT_BE_IN_WORKER" not in captured_env.values()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_worker_skips_r2_credentials_when_ro_key_is_missing(monkeypatch):
    """Worker env should omit VISGATE_R2_* when no platform RO key is configured."""
    from src.core.config import get_settings
    get_settings.cache_clear()
    monkeypatch.delenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_INPUT_R", raising=False)
    monkeypatch.delenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_INPUT_R", raising=False)
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
            hf_token_override="hf_user_token",
        )

    assert "VISGATE_R2_ACCESS_KEY_ID" not in captured_env
    assert "VISGATE_R2_SECRET_ACCESS_KEY" not in captured_env
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_vast_cache_hit_sets_s3_model_url_with_ro_fallback(monkeypatch):
    """Vast deployments should still get S3_MODEL_URL when RW key is missing but RO key exists."""
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VAST_API_KEY", "vast_test_key")
    monkeypatch.delenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", raising=False)
    monkeypatch.delenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", raising=False)
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_INPUT_R", "ro_key_EXPECTED")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_INPUT_R", "ro_secret_EXPECTED")
    get_settings.cache_clear()

    captured_env: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_env.update(kwargs.get("env", {}))
        return {"id": "ep-vast-1", "url": "vast-ep://ep-vast-1"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)
    mock_provider.check_endpoint_health = AsyncMock(return_value={"status": "ready", "running_count": 1})

    vast_doc = _make_deployment_doc()
    vast_doc.provider = "vast"

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.get_deployment", return_value=vast_doc))
        stack.enter_context(patch("src.services.deployment.fetch_cached_model_ids", return_value=["stabilityai/sd-turbo"]))
        stack.enter_context(patch("src.services.deployment.mark_deployment_ready_and_notify", new_callable=AsyncMock, return_value=True))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_vast_cache",
            runpod_api_key="rpa_userkey",
            hf_token_override="hf_user_token",
        )

    assert captured_env.get("S3_MODEL_URL") == "s3://visgate-models/models/stabilityai--sd-turbo"
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
            hf_token_override="hf_user_token",
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
            hf_token_override="hf_user_token",
        )

    assert captured_env.get("VISGATE_WEBHOOK") == "https://api.example.com/internal/deployment-ready/dep_cleanup"
    assert captured_env.get("VISGATE_LOG_TUNNEL") == "https://api.example.com/internal/logs/dep_cleanup"
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# HF token policy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_hf_token_fails_deployment(monkeypatch):
    """Deployments must fail fast when caller omits HF token."""
    from src.core.config import get_settings
    get_settings.cache_clear()

    update_mock = MagicMock()
    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock()

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.update_deployment", update_mock))
        from src.services.deployment import orchestrate_deployment
        await orchestrate_deployment(
            deployment_id="dep_hf",
            runpod_api_key="rpa_key",
            hf_token_override=None,
        )

    assert any(call.args[3].get("error") == "Missing Hugging Face token" for call in update_mock.call_args_list)
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_caller_hf_token_is_injected_into_worker(monkeypatch):
    """Caller-provided hf_token is forwarded into worker env."""
    from src.core.config import get_settings
    get_settings.cache_clear()
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


def test_as_run_url_keeps_vast_virtual_url() -> None:
    from src.services.deployment import _as_run_url

    assert _as_run_url("vast-ep://my-endpoint") == "vast-ep://my-endpoint"

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
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, return_value={"id": "tpl-audio"}))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_audio",
            runpod_api_key="rpa_audio",
            hf_token_override="hf_user_token",
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
    monkeypatch.setenv("RUNPOD_WORKERS_MIN", "1")
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
            hf_token_override="hf_user_token",
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
            hf_token_override="hf_user_token",
        )

    probe_mock.assert_not_awaited()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_video_deployment_uses_r2_cache_hit_s3_url(monkeypatch):
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID_VIDEO", "tpl-video")
    monkeypatch.setenv("DOCKER_IMAGE_VIDEO", "visgateai/inference-video:latest")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", "rw_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", "rw_secret")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_INPUT_R", "ro_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_INPUT_R", "ro_secret")
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
            hf_token_override="hf_user_token",
        )

    # Env is now baked into the per-deployment RunPod template, not the endpoint.
    assert captured_kwargs["env"] == {}
    tpl_env = {e["key"]: e["value"] for e in sync_template_mock_r2.call_args.kwargs["env"]}
    # R2 cache is now enabled for video; on a cache HIT the worker gets S3_MODEL_URL
    assert "S3_MODEL_URL" in tpl_env
    assert tpl_env["HF_MODEL_ID"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_vast_failed_status_triggers_endpoint_cleanup(monkeypatch):
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VAST_API_KEY", "vast_test_key")
    get_settings.cache_clear()

    initial_doc = _make_deployment_doc()
    initial_doc.provider = "vast"
    failed_doc = _make_deployment_doc(status="failed")
    failed_doc.provider = "vast"

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(return_value={"id": "14199", "url": "vast-ep://dep"})
    mock_provider.delete_endpoint = AsyncMock(return_value=None)
    mock_provider.check_endpoint_health = AsyncMock(return_value={"status": "loading", "running_count": 0})

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                side_effect=[initial_doc, failed_doc],
            )
        )
        stack.enter_context(patch("src.services.gpu_registry.fetch_live_gpu_registry", new_callable=AsyncMock, return_value=[]))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_vast_cleanup",
            runpod_api_key="rpa_user_key",
            hf_token_override="hf_user_token",
        )

    mock_provider.delete_endpoint.assert_awaited_once_with("14199", "vast_test_key")
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_vast_create_endpoint_uses_nonzero_cold_workers(monkeypatch):
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VAST_API_KEY", "vast_test_key")
    monkeypatch.setenv("RUNPOD_WORKERS_MIN", "1")
    monkeypatch.setenv("RUNPOD_WORKERS_MAX", "1")
    get_settings.cache_clear()

    initial_doc = _make_deployment_doc()
    initial_doc.provider = "vast"
    ready_doc = _make_deployment_doc(status="ready")
    ready_doc.provider = "vast"

    captured_kwargs: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_kwargs.update(kwargs)
        return {"id": "14200", "url": "vast-ep://dep"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)
    mock_provider.delete_endpoint = AsyncMock(return_value=None)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                side_effect=[initial_doc, ready_doc],
            )
        )
        stack.enter_context(patch("src.services.gpu_registry.fetch_live_gpu_registry", new_callable=AsyncMock, return_value=[]))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_vast_cold_workers",
            runpod_api_key="rpa_user_key",
            hf_token_override="hf_user_token",
        )

    assert captured_kwargs.get("cold_workers") == 1
    assert captured_kwargs.get("max_workers") == 1
    assert captured_kwargs.get("gpu_id") == "8"
    assert captured_kwargs.get("gpu_ids") == ["GPU_A40"]
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_vast_deployment_uses_r2_cache(monkeypatch):
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VAST_API_KEY", "vast_test_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", "rw_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", "rw_secret")
    get_settings.cache_clear()

    initial_doc = _make_deployment_doc()
    initial_doc.provider = "vast"
    ready_doc = _make_deployment_doc(status="ready")
    ready_doc.provider = "vast"

    captured_kwargs: dict = {}

    async def mock_create_endpoint(**kwargs):
        captured_kwargs.update(kwargs)
        return {"id": "14202", "url": "vast-ep://dep"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(
            patch(
                "src.services.deployment.get_deployment",
                side_effect=[initial_doc, ready_doc],
            )
        )
        stack.enter_context(patch("src.services.gpu_registry.fetch_live_gpu_registry", new_callable=AsyncMock, return_value=[]))
        fetch_cached_mock = stack.enter_context(patch("src.services.deployment.fetch_cached_model_ids", return_value=["stabilityai/sd-turbo"]))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_vast_no_cache",
            runpod_api_key="rpa_user_key",
            hf_token_override="hf_user_token",
        )

    fetch_cached_mock.assert_called_once()
    assert captured_kwargs.get("env", {}).get("S3_MODEL_URL") == "s3://visgate-models/models/stabilityai--sd-turbo"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_manifest_miss_uses_direct_r2_prefix_fallback(monkeypatch):
    """If manifest misses model entry but prefix exists, deployment should still inject S3_MODEL_URL."""
    from src.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_OUTPUT_RW", "rw_key")
    monkeypatch.setenv("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_OUTPUT_RW", "rw_secret")
    get_settings.cache_clear()

    captured_env, template_capturer = _make_template_capturer()

    async def mock_create_endpoint(**kwargs):
        return {"id": "ep-fallback", "url": "https://api.runpod.ai/v2/ep-fallback/run"}

    mock_provider = MagicMock()
    mock_provider.create_endpoint = AsyncMock(side_effect=mock_create_endpoint)

    with ExitStack() as stack:
        _apply_base_patches(stack, mock_provider)
        stack.enter_context(patch("src.services.deployment.fetch_cached_model_ids", return_value=set()))
        stack.enter_context(patch("src.services.deployment.model_cached_in_bucket", return_value=True))
        stack.enter_context(patch("src.services.deployment.create_serverless_template", new_callable=AsyncMock, side_effect=template_capturer))

        from src.services.deployment import orchestrate_deployment

        await orchestrate_deployment(
            deployment_id="dep_manifest_fallback",
            runpod_api_key="rpa_key",
            hf_token_override="hf_user_token",
        )

    assert captured_env.get("S3_MODEL_URL", "").endswith("/stabilityai--sd-turbo")
    get_settings.cache_clear()

