"""Unit tests for Vast.ai On-Demand Instance provider (src/services/vast.py)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from src.core.errors import VastAPIError
from src.services.vast import VastProvider, _JOB_ID_SEP, _build_search_params


@pytest.fixture
def provider() -> VastProvider:
    return VastProvider(console_base="https://console.vast.ai")


def _mock_response(status_code: int = 200, json_data: Any = None, text: str = "") -> httpx.Response:
    """Build a fake httpx.Response for testing."""
    if json_data is not None:
        content = json.dumps(json_data).encode()
        headers = {"content-type": "application/json"}
    else:
        content = text.encode()
        headers = {"content-type": "text/plain"}
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers,
        request=httpx.Request("GET", "https://example.com"),
    )


# ── URL/ID helpers ───────────────────────────────────────────────────────────

def test_build_and_parse_endpoint_url() -> None:
    url = VastProvider.build_endpoint_url("14252")
    assert url == "vast-inst://14252"
    assert VastProvider.parse_endpoint_id(url) == "14252"


def test_parse_endpoint_id_plain() -> None:
    assert VastProvider.parse_endpoint_id("12345") == "12345"


def test_parse_endpoint_id_legacy_vast_ep() -> None:
    """Backward compat: parse_endpoint_id accepts old vast-ep:// format too."""
    assert VastProvider.parse_endpoint_id("vast-ep://42/ep1") == "42"


def test_encode_decode_job_id() -> None:
    composite = VastProvider.encode_job_id("http://1.2.3.4:8000", "abc-123")
    assert _JOB_ID_SEP in composite
    worker_url, actual_id = VastProvider.decode_job_id(composite)
    assert worker_url == "http://1.2.3.4:8000"
    assert actual_id == "abc-123"


def test_decode_job_id_no_separator() -> None:
    worker_url, actual_id = VastProvider.decode_job_id("plain-id")
    assert worker_url == ""
    assert actual_id == "plain-id"


def test_extract_worker_url() -> None:
    inst = {"public_ipaddr": "1.2.3.4", "ports": {"8000/tcp": [{"HostPort": "45678"}]}}
    assert VastProvider.extract_worker_url(inst) == "http://1.2.3.4:45678"


def test_extract_worker_url_no_ip() -> None:
    assert VastProvider.extract_worker_url({"ports": {}}) is None


def test_extract_worker_url_no_port() -> None:
    assert VastProvider.extract_worker_url({"public_ipaddr": "1.2.3.4", "ports": {}}) is None


# ── _request ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_uses_bearer_auth(provider: VastProvider) -> None:
    """Verify requests use Authorization: Bearer header (not query param)."""
    captured_request = None

    async def mock_send(self, request, **kwargs):
        nonlocal captured_request
        captured_request = request
        return _mock_response(200, {"ok": True})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        await provider._request("GET", "/api/v0/test/", "my-secret-key")

    assert captured_request is not None
    assert captured_request.headers.get("authorization") == "Bearer my-secret-key"
    assert "api_key" not in str(captured_request.url)


@pytest.mark.asyncio
async def test_request_retries_on_500(provider: VastProvider) -> None:
    call_count = 0

    async def mock_send(self, request, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return _mock_response(500, text="Server Error")
        return _mock_response(200, {"ok": True})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider._request("GET", "/test/", "key", _max_retries=3)

    assert result == {"ok": True}
    assert call_count == 3


@pytest.mark.asyncio
async def test_request_raises_on_4xx(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(403, text="Forbidden")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="403"):
            await provider._request("GET", "/api/v0/test/", "bad-key")


# ── search offers ────────────────────────────────────────────────────────────

def test_build_search_params_includes_gpu_pool() -> None:
    assert _build_search_params(24, ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000"]) == (
        "verified=true rentable=true rented=false gpu_ram>=24 gpu_name in [RTX_3090, RTX_A5000]"
    )


@pytest.mark.asyncio
async def test_search_offers(provider: VastProvider) -> None:
    offers = [{"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.5}]

    async def mock_send(self, request, **kwargs):
        assert "/api/v0/search/offers/" in str(request.url)
        return _mock_response(200, {"offers": offers})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.search_offers("key", gpu_ram=24000)

    assert len(result) == 1
    assert result[0]["gpu_name"] == "RTX 4090"


# ── create_endpoint (on-demand instance flow) ────────────────────────────────

@pytest.mark.asyncio
async def test_create_endpoint_success(provider: VastProvider) -> None:
    offers_resp = {"offers": [
        {"id": 100, "gpu_name": "RTX 4090", "dph_total": 0.45},
        {"id": 200, "gpu_name": "RTX 4090", "dph_total": 0.55},
    ]}
    instance_resp = {"success": True, "new_contract": 9999}

    call_log: list[str] = []

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/search/offers/" in url:
            call_log.append("search")
            return _mock_response(200, offers_resp)
        if "/api/v0/asks/100/" in url:
            call_log.append("create_instance")
            body = json.loads(request.content)
            assert body["image"] == "visgateai/inference-image:latest"
            assert body["runtype"] == "args"
            assert "-p 8000:8000" in body["env"]
            return _mock_response(200, instance_resp)
        return _mock_response(404, text="not found")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        ep = await provider.create_endpoint(
            name="ep1",
            gpu_id="24",
            image="visgateai/inference-image:latest",
            env={"HF_MODEL_ID": "stabilityai/sd-turbo"},
            api_key="test-key",
            gpu_ids=["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"],
        )

    assert call_log == ["search", "create_instance"]
    assert ep["id"] == "9999"
    assert ep["url"] == "vast-inst://9999"
    assert ep["raw_response"]["offer_id"] == 100
    assert ep["raw_response"]["gpu_name"] == "RTX 4090"


@pytest.mark.asyncio
async def test_create_endpoint_no_offers(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {"offers": []})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="No matching GPU offers"):
            await provider.create_endpoint(
                name="ep1", gpu_id="24", image="img:latest", env={}, api_key="key",
            )


@pytest.mark.asyncio
async def test_create_endpoint_no_contract_id(provider: VastProvider) -> None:
    offers_resp = {"offers": [{"id": 100, "gpu_name": "RTX 4090", "dph_total": 0.5}]}
    instance_resp = {"success": True}  # no new_contract

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/search/offers/" in url:
            return _mock_response(200, offers_resp)
        return _mock_response(200, instance_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="no contract ID"):
            await provider.create_endpoint(
                name="ep1", gpu_id="24", image="img:latest", env={}, api_key="key",
            )


# ── delete_endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_endpoint(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        assert "/api/v0/instances/42/" in str(request.url)
        assert request.method == "DELETE"
        return _mock_response(200, {"success": True})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        await provider.delete_endpoint("42", "test-key")


# ── list_endpoints ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_endpoints(provider: VastProvider) -> None:
    resp = {
        "instances": [
            {"id": 10, "label": "visgate-alpha", "actual_status": "running"},
            {"id": 11, "label": "visgate-beta", "actual_status": "loading"},
        ],
    }

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        summaries = await provider.list_endpoints("test-key")

    assert len(summaries) == 2
    assert summaries[0]["id"] == "10"
    assert summaries[0]["name"] == "visgate-alpha"
    assert summaries[0]["status"] == "running"
    assert summaries[0]["url"] == "vast-inst://10"


# ── list_gpu_types ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_gpu_types(provider: VastProvider) -> None:
    resp = {
        "offers": [
            {"id": 1, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.45},
            {"id": 2, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.50},
            {"id": 3, "gpu_name": "A100", "gpu_ram": 81920, "dph_total": 1.20},
        ],
    }

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        gpus = await provider.list_gpu_types("test-key")

    assert len(gpus) == 2
    names = {g["displayName"] for g in gpus}
    assert "RTX 4090" in names
    assert "A100" in names


# ── submit_job ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_job_with_direct_url(provider: VastProvider) -> None:
    """submit_job with an HTTP URL goes directly to the worker."""
    job_resp = {"id": "job-42", "status": "IN_QUEUE"}

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        assert "10.0.0.1" in url
        assert "/run" in url
        return _mock_response(200, job_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        accepted = await provider.submit_job(
            "http://10.0.0.1:8000",
            "api-key",
            {"prompt": "a cat"},
        )

    assert _JOB_ID_SEP in accepted["id"]
    worker_url, actual_id = VastProvider.decode_job_id(accepted["id"])
    assert worker_url == "http://10.0.0.1:8000"
    assert actual_id == "job-42"
    assert accepted["status"] == "IN_QUEUE"


@pytest.mark.asyncio
async def test_submit_job_via_virtual_url(provider: VastProvider) -> None:
    """submit_job with vast-inst:// looks up instance to get IP:port."""
    instance_resp = {"actual_status": "running", "public_ipaddr": "5.6.7.8", "ports": {"8000/tcp": [{"HostPort": "12345"}]}}
    job_resp = {"id": "job-99", "status": "IN_QUEUE"}

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/instances/42/" in url:
            return _mock_response(200, instance_resp)
        if "/run" in url:
            assert "5.6.7.8:12345" in url
            return _mock_response(200, job_resp)
        return _mock_response(404, text="not found")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        accepted = await provider.submit_job(
            "vast-inst://42",
            "api-key",
            {"prompt": "a cat"},
        )

    assert accepted["status"] == "IN_QUEUE"


@pytest.mark.asyncio
async def test_submit_job_no_public_ip(provider: VastProvider) -> None:
    instance_resp = {"actual_status": "loading", "ports": {}}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, instance_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="no public IP"):
            await provider.submit_job("vast-inst://42", "key", {})


@pytest.mark.asyncio
async def test_submit_job_worker_error(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(500, text="Internal Server Error")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="500"):
            await provider.submit_job("http://10.0.0.1:8000", "key", {"prompt": "x"})


# ── get_job_status ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_status_completed(provider: VastProvider) -> None:
    status_resp = {"id": "job-42", "status": "COMPLETED", "output": {"url": "http://example.com/out.png"}}
    composite_id = VastProvider.encode_job_id("http://10.0.0.1:8000", "job-42")

    async def mock_send(self, request, **kwargs):
        assert "10.0.0.1" in str(request.url)
        assert "/status/job-42" in str(request.url)
        return _mock_response(200, status_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        status = await provider.get_job_status("vast-inst://42", composite_id, "key")

    assert status["status"] == "COMPLETED"
    assert status["output"]["url"] == "http://example.com/out.png"


@pytest.mark.asyncio
async def test_get_job_status_no_worker_url(provider: VastProvider) -> None:
    with pytest.raises(VastAPIError, match="Cannot determine worker URL"):
        await provider.get_job_status("vast-inst://42", "plain-id-no-sep", "key")


# ── check_endpoint_health ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_endpoint_health_running_and_ready(provider: VastProvider) -> None:
    """Instance running with a ready model returns status=ready with worker_url."""
    instance_resp = {
        "actual_status": "running",
        "public_ipaddr": "1.2.3.4",
        "ports": {"8000/tcp": [{"HostPort": "45678"}]},
    }
    health_resp = {"workers": {"ready": 1, "idle": 0, "initializing": 0}}

    call_log: list[str] = []

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/instances/42/" in url:
            call_log.append("instance")
            return _mock_response(200, instance_resp)
        if "1.2.3.4:45678/health" in url:
            call_log.append("health_probe")
            return _mock_response(200, health_resp)
        return _mock_response(404, text="not found")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "ready"
    assert health["worker_url"] == "http://1.2.3.4:45678"
    assert health["running_count"] == 1
    assert "instance" in call_log
    assert "health_probe" in call_log


@pytest.mark.asyncio
async def test_check_endpoint_health_loading(provider: VastProvider) -> None:
    """Instance in loading state returns loading."""
    instance_resp = {"actual_status": "loading"}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, instance_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "loading"
    assert health["running_count"] == 0


@pytest.mark.asyncio
async def test_check_endpoint_health_running_no_port(provider: VastProvider) -> None:
    """Instance running but no port exposed yet."""
    instance_resp = {"actual_status": "running", "ports": {}}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, instance_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "loading"


@pytest.mark.asyncio
async def test_check_endpoint_health_model_loading(provider: VastProvider) -> None:
    """Instance running, server up, but model not loaded yet."""
    instance_resp = {
        "actual_status": "running",
        "public_ipaddr": "1.2.3.4",
        "ports": {"8000/tcp": [{"HostPort": "45678"}]},
    }
    health_resp = {"workers": {"ready": 0, "idle": 0, "initializing": 1}}

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/instances/42/" in url:
            return _mock_response(200, instance_resp)
        if "/health" in url:
            return _mock_response(200, health_resp)
        return _mock_response(404, text="not found")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "loading"
    assert health["worker_url"] == "http://1.2.3.4:45678"


@pytest.mark.asyncio
async def test_check_endpoint_health_api_error(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(401, text="Unauthorized")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "error"


@pytest.mark.asyncio
async def test_check_endpoint_health_error_msg(provider: VastProvider) -> None:
    """Vast API returning an error_msg field should surface as error."""
    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {"error_msg": "instance not found"})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "error"
    assert "instance not found" in health["error"]


# ── get_run_url ──────────────────────────────────────────────────────────────

def test_get_run_url(provider: VastProvider) -> None:
    assert provider.get_run_url("12345") == "vast-inst://12345"


# ── provider_factory registration ────────────────────────────────────────────

def test_vast_registered_in_factory() -> None:
    from src.services.provider_factory import get_provider

    p = get_provider("vast")
    assert isinstance(p, VastProvider)
