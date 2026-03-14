"""Unit tests for Vast.ai provider (src/services/vast.py)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.core.errors import VastAPIError
from src.services.vast import VastProvider


@pytest.fixture
def provider() -> VastProvider:
    return VastProvider(api_base="https://cloud.vast.ai/api/v0")


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


# ── search_offers ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_offers_success(provider: VastProvider) -> None:
    offers = [
        {"id": 1, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.45},
        {"id": 2, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.35},
    ]

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {"offers": offers})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.search_offers("test-key", min_gpu_ram_gb=24)

    assert len(result) == 2
    assert result[0]["gpu_name"] == "RTX 4090"


@pytest.mark.asyncio
async def test_search_offers_empty(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {"offers": []})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.search_offers("test-key", min_gpu_ram_gb=80)

    assert result == []


@pytest.mark.asyncio
async def test_search_offers_api_error(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(403, text="Forbidden")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="403"):
            await provider.search_offers("bad-key")


# ── create_endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_endpoint_success(provider: VastProvider) -> None:
    offers = [{"id": 42, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.45}]
    create_resp = {"success": True, "new_contract": 999}
    running_instance = {
        "id": 999,
        "actual_status": "running",
        "public_ipaddr": "1.2.3.4",
        "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "54321"}]},
    }

    call_count = 0

    async def mock_send(self, request, **kwargs):
        nonlocal call_count
        call_count += 1
        url = str(request.url)
        if "/bundles/" in url:
            return _mock_response(200, {"offers": offers})
        if "/asks/" in url:
            return _mock_response(200, create_resp)
        if "/instances/999/" in url:
            return _mock_response(200, running_instance)
        return _mock_response(404, text="not found")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        ep = await provider.create_endpoint(
            name="test-ep",
            gpu_id="24",
            image="visgateai/inference-image:latest",
            env={"HF_MODEL_ID": "stabilityai/sd-turbo"},
            api_key="test-key",
        )

    assert ep["id"] == "999"
    assert ep["url"] == "http://1.2.3.4:54321"


@pytest.mark.asyncio
async def test_create_endpoint_no_offers(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {"offers": []})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="No Vast.ai offers found"):
            await provider.create_endpoint(
                name="test-ep",
                gpu_id="24",
                image="visgateai/inference-image:latest",
                env={},
                api_key="test-key",
            )


# ── delete_endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_endpoint(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        await provider.delete_endpoint("999", "test-key")


# ── list_endpoints ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_endpoints(provider: VastProvider) -> None:
    instances = [
        {"id": 100, "actual_status": "running", "label": "ep-1", "public_ipaddr": "1.1.1.1", "ports": {}},
        {"id": 101, "actual_status": "loading", "label": "ep-2", "public_ipaddr": None, "ports": {}},
    ]

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {"instances": instances})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        summaries = await provider.list_endpoints("test-key")

    assert len(summaries) == 2
    assert summaries[0]["id"] == "100"
    assert summaries[0]["name"] == "ep-1"
    assert summaries[0]["status"] == "running"


# ── list_gpu_types ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_gpu_types(provider: VastProvider) -> None:
    offers = [
        {"id": 1, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.45},
        {"id": 2, "gpu_name": "RTX 4090", "gpu_ram": 24576, "dph_total": 0.50},
        {"id": 3, "gpu_name": "A100", "gpu_ram": 81920, "dph_total": 1.20},
    ]

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, {"offers": offers})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        gpus = await provider.list_gpu_types("test-key")

    assert len(gpus) == 2  # deduplicated
    names = {g["displayName"] for g in gpus}
    assert "RTX 4090" in names
    assert "A100" in names


# ── submit_job ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_job_success(provider: VastProvider) -> None:
    job_resp = {"id": "abc-123", "status": "IN_QUEUE"}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, job_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        accepted = await provider.submit_job(
            "http://1.2.3.4:54321",
            "api-key",
            {"prompt": "a cat"},
        )

    assert accepted["id"] == "abc-123"
    assert accepted["status"] == "IN_QUEUE"


@pytest.mark.asyncio
async def test_submit_job_error(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(500, text="Internal Server Error")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="500"):
            await provider.submit_job("http://1.2.3.4:54321", "api-key", {})


# ── get_job_status ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_status_completed(provider: VastProvider) -> None:
    status_resp = {"id": "abc-123", "status": "COMPLETED", "output": {"url": "http://example.com/out.png"}}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, status_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        status = await provider.get_job_status("http://1.2.3.4:54321", "abc-123", "api-key")

    assert status["status"] == "COMPLETED"
    assert status["output"]["url"] == "http://example.com/out.png"


@pytest.mark.asyncio
async def test_get_job_status_not_found(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(404, {"id": "bad", "status": "NOT_FOUND", "error": "Job not found"})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="404"):
            await provider.get_job_status("http://1.2.3.4:54321", "bad", "api-key")


# ── get_run_url ──────────────────────────────────────────────────────────────

def test_get_run_url(provider: VastProvider) -> None:
    assert provider.get_run_url("12345") == "vast://12345/run"


# ── _instance_endpoint_url ───────────────────────────────────────────────────

def test_instance_endpoint_url_with_ports(provider: VastProvider) -> None:
    inst = {
        "public_ipaddr": "10.0.0.1",
        "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9999"}]},
    }
    assert provider._instance_endpoint_url(inst) == "http://10.0.0.1:9999"


def test_instance_endpoint_url_direct_port(provider: VastProvider) -> None:
    inst = {"public_ipaddr": "10.0.0.1", "ports": {}, "direct_port_start": 40000}
    assert provider._instance_endpoint_url(inst) == "http://10.0.0.1:40000"


def test_instance_endpoint_url_missing(provider: VastProvider) -> None:
    inst = {"ports": {}}
    assert provider._instance_endpoint_url(inst) is None


# ── provider_factory registration ────────────────────────────────────────────

def test_vast_registered_in_factory() -> None:
    from src.services.provider_factory import get_provider

    p = get_provider("vast")
    assert isinstance(p, VastProvider)


# ── retry on transient errors ────────────────────────────────────────────────

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
