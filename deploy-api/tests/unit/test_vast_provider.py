"""Unit tests for Vast.ai Serverless provider (src/services/vast.py)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.core.errors import VastAPIError
from src.services.vast import VastProvider, _JOB_ID_SEP, _build_search_params


@pytest.fixture
def provider() -> VastProvider:
    return VastProvider(
        console_base="https://console.vast.ai",
        route_base="https://run.vast.ai",
    )


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
    assert url == "vast-ep://14252"
    assert VastProvider.parse_endpoint_id(url) == "14252"


def test_parse_endpoint_id_plain() -> None:
    assert VastProvider.parse_endpoint_id("12345") == "12345"


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


def test_env_dict_to_flag_str() -> None:
    env = {"A": "1", "B": "2"}
    result = VastProvider._env_dict_to_flag_str(env)
    assert "-e A=1" in result
    assert "-e B=2" in result
    assert "-p 8000:8000" in result


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
    # No api_key in query params
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


# ── create_template ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_template(provider: VastProvider) -> None:
    resp = {"success": True, "template": {"id": 100, "hash_id": "abc123", "name": "test"}}

    async def mock_send(self, request, **kwargs):
        assert "/api/v0/template/" in str(request.url)
        body = json.loads(request.content)
        assert body["name"] == "visgate-test"
        assert body["image"] == "visgateai/inference-image:latest"
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.create_template(
            "key", name="visgate-test", image="visgateai/inference-image:latest", env_str="-e A=1"
        )

    assert result["template"]["hash_id"] == "abc123"


# ── create_serverless_endpoint ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_serverless_endpoint(provider: VastProvider) -> None:
    resp = {"success": True, "result": 42}

    async def mock_send(self, request, **kwargs):
        assert "/api/v0/endptjobs/" in str(request.url)
        body = json.loads(request.content)
        assert body["endpoint_name"] == "my-ep"
        assert body["max_workers"] == 2
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.create_serverless_endpoint(
            "key", endpoint_name="my-ep", max_workers=2,
        )

    assert result["result"] == 42


# ── create_workergroup ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_workergroup(provider: VastProvider) -> None:
    resp = {"success": True, "id": 789}

    async def mock_send(self, request, **kwargs):
        assert "/api/v0/workergroups/" in str(request.url)
        body = json.loads(request.content)
        assert body["endpoint_name"] == "my-ep"
        assert body["template_hash"] == "abc123"
        assert body["gpu_ram"] == 24
        assert body["search_params"] == "verified=true rentable=true rented=false gpu_ram>=24"
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.create_workergroup(
            "key", endpoint_name="my-ep", template_hash="abc123", gpu_ram=24,
        )

    assert result["id"] == 789


def test_build_search_params_includes_gpu_pool() -> None:
    assert _build_search_params(24, ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000"]) == (
        "verified=true rentable=true rented=false gpu_ram>=24 gpu_name in [RTX_3090, RTX_A5000]"
    )


# ── create_endpoint (full flow) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_endpoint_success(provider: VastProvider) -> None:
    template_resp = {"success": True, "template": {"id": 100, "hash_id": "tpl_hash", "name": "visgate-ep1"}}
    endpoint_resp = {"success": True, "result": 42}
    wg_resp = {"success": True, "id": 789}

    call_log: list[str] = []

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/template/" in url:
            call_log.append("template")
            return _mock_response(200, template_resp)
        if "/api/v0/endptjobs/" in url:
            call_log.append("endpoint")
            return _mock_response(200, endpoint_resp)
        if "/api/v0/workergroups/" in url:
            call_log.append("workergroup")
            body = json.loads(request.content)
            assert body["gpu_ram"] == 24
            assert body["search_params"].endswith("gpu_name in [RTX_A5000, RTX_3090]")
            return _mock_response(200, wg_resp)
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

    assert call_log == ["template", "endpoint", "workergroup"]
    assert ep["id"] == "42"
    assert ep["url"] == "vast-ep://42/ep1"
    assert ep["raw_response"]["template_hash"] == "tpl_hash"
    assert ep["raw_response"]["workergroup_id"] == 789


@pytest.mark.asyncio
async def test_create_endpoint_no_template_hash(provider: VastProvider) -> None:
    template_resp = {"success": True, "template": {"id": 100, "name": "x"}}  # no hash_id

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, template_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="hash_id"):
            await provider.create_endpoint(
                name="ep1", gpu_id="24", image="img:latest", env={}, api_key="key",
            )


@pytest.mark.asyncio
async def test_create_endpoint_no_endpoint_id(provider: VastProvider) -> None:
    template_resp = {"success": True, "template": {"id": 1, "hash_id": "h", "name": "n"}}
    endpoint_resp = {"success": True}  # no result field

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/template/" in url:
            return _mock_response(200, template_resp)
        return _mock_response(200, endpoint_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="no ID"):
            await provider.create_endpoint(
                name="ep1", gpu_id="24", image="img:latest", env={}, api_key="key",
            )


@pytest.mark.asyncio
async def test_create_endpoint_workergroup_failure_cleans_up_endpoint(provider: VastProvider) -> None:
    template_resp = {"success": True, "template": {"id": 1, "hash_id": "h", "name": "n"}}
    endpoint_resp = {"success": True, "result": 42}

    call_log: list[tuple[str, str]] = []

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/api/v0/template/" in url:
            call_log.append(("template", request.method))
            return _mock_response(200, template_resp)
        if "/api/v0/endptjobs/" in url and request.method == "POST":
            call_log.append(("endpoint_create", request.method))
            return _mock_response(200, endpoint_resp)
        if "/api/v0/workergroups/" in url:
            call_log.append(("workergroup", request.method))
            return _mock_response(500, text="wg failed")
        if "/api/v0/endptjobs/42/" in url and request.method == "DELETE":
            call_log.append(("endpoint_delete", request.method))
            return _mock_response(200, {"success": True})
        return _mock_response(404, text="not found")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="500"):
            await provider.create_endpoint(
                name="ep1", gpu_id="24", image="img:latest", env={}, api_key="key",
            )

    assert call_log[0] == ("template", "POST")
    assert call_log[1] == ("endpoint_create", "POST")
    assert ("workergroup", "POST") in call_log
    assert call_log[-1] == ("endpoint_delete", "DELETE")


# ── delete_endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_endpoint(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        assert "/api/v0/endptjobs/42/" in str(request.url)
        assert request.method == "DELETE"
        return _mock_response(200, {"success": True})

    with patch.object(httpx.AsyncClient, "send", mock_send):
        await provider.delete_endpoint("42", "test-key")


# ── list_endpoints ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_endpoints(provider: VastProvider) -> None:
    resp = {
        "success": True,
        "results": [
            {"id": 10, "endpoint_name": "ep-alpha", "endpoint_state": "active"},
            {"id": 11, "endpoint_name": "ep-beta", "endpoint_state": "stopped"},
        ],
    }

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        summaries = await provider.list_endpoints("test-key")

    assert len(summaries) == 2
    assert summaries[0]["id"] == "10"
    assert summaries[0]["name"] == "ep-alpha"
    assert summaries[0]["status"] == "active"
    assert summaries[0]["url"] == "vast-ep://10/ep-alpha"


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


# ── route_request ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_request_success(provider: VastProvider) -> None:
    resp = {"url": "http://10.0.0.1:8000", "reqnum": 1, "signature": "sig"}

    async def mock_send(self, request, **kwargs):
        assert "run.vast.ai" in str(request.url)
        assert "/route/" in str(request.url)
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.route_request("key", "my-endpoint")

    assert result["url"] == "http://10.0.0.1:8000"


@pytest.mark.asyncio
async def test_route_request_stopped(provider: VastProvider) -> None:
    resp = {"status": "Stopped"}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.route_request("key", "my-endpoint")

    assert result["status"] == "Stopped"


# ── submit_job ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_job_success(provider: VastProvider) -> None:
    route_resp = {"url": "http://10.0.0.1:8000", "reqnum": 1, "signature": "sig"}
    job_resp = {"id": "job-42", "status": "IN_QUEUE"}

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/route/" in url:
            return _mock_response(200, route_resp)
        if "/run" in url:
            assert "10.0.0.1" in url
            return _mock_response(200, job_resp)
        return _mock_response(404, text="not found")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        accepted = await provider.submit_job(
            "vast-ep://my-endpoint",
            "api-key",
            {"prompt": "a cat"},
        )

    assert _JOB_ID_SEP in accepted["id"]
    worker_url, actual_id = VastProvider.decode_job_id(accepted["id"])
    assert worker_url == "http://10.0.0.1:8000"
    assert actual_id == "job-42"
    assert accepted["status"] == "IN_QUEUE"


@pytest.mark.asyncio
async def test_submit_job_no_workers(provider: VastProvider) -> None:
    route_resp = {"status": "Stopped"}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, route_resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="stopped"):
            await provider.submit_job("vast-ep://my-endpoint", "key", {})


@pytest.mark.asyncio
async def test_submit_job_worker_error(provider: VastProvider) -> None:
    route_resp = {"url": "http://10.0.0.1:8000", "reqnum": 1, "signature": "sig"}

    async def mock_send(self, request, **kwargs):
        url = str(request.url)
        if "/route/" in url:
            return _mock_response(200, route_resp)
        return _mock_response(500, text="Internal Server Error")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        with pytest.raises(VastAPIError, match="500"):
            await provider.submit_job("vast-ep://my-endpoint", "key", {"prompt": "x"})


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
        status = await provider.get_job_status("vast-ep://my-endpoint", composite_id, "key")

    assert status["status"] == "COMPLETED"
    assert status["output"]["url"] == "http://example.com/out.png"


@pytest.mark.asyncio
async def test_get_job_status_no_worker_url(provider: VastProvider) -> None:
    with pytest.raises(VastAPIError, match="Cannot determine worker URL"):
        await provider.get_job_status("vast-ep://ep", "plain-id-no-sep", "key")


# ── get_endpoint_workers ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_endpoint_workers(provider: VastProvider) -> None:
    resp = {"workers": [{"id": 1, "status": "ready", "url": "http://10.0.0.1:8000"}]}

    async def mock_send(self, request, **kwargs):
        assert "run.vast.ai" in str(request.url)
        assert "/get_endpoint_workers/" in str(request.url)
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        result = await provider.get_endpoint_workers("key", 42)

    assert len(result["workers"]) == 1
    assert result["workers"][0]["status"] == "ready"


# ── check_endpoint_health ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_endpoint_health_running(provider: VastProvider) -> None:
    resp = {"workers": [{"id": 1, "status": "ready"}, {"id": 2, "status": "loading"}]}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "ready"
    assert health["running_count"] == 1
    assert health["total_count"] == 2


@pytest.mark.asyncio
async def test_check_endpoint_health_no_workers(provider: VastProvider) -> None:
    resp = {"workers": []}

    async def mock_send(self, request, **kwargs):
        return _mock_response(200, resp)

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "no_workers"
    assert health["running_count"] == 0


@pytest.mark.asyncio
async def test_check_endpoint_health_api_error(provider: VastProvider) -> None:
    async def mock_send(self, request, **kwargs):
        return _mock_response(401, text="Unauthorized")

    with patch.object(httpx.AsyncClient, "send", mock_send):
        health = await provider.check_endpoint_health("42", "key")

    assert health["status"] == "error"


# ── get_run_url ──────────────────────────────────────────────────────────────

def test_get_run_url(provider: VastProvider) -> None:
    assert provider.get_run_url("12345") == "https://run.vast.ai/route/"


# ── provider_factory registration ────────────────────────────────────────────

def test_vast_registered_in_factory() -> None:
    from src.services.provider_factory import get_provider

    p = get_provider("vast")
    assert isinstance(p, VastProvider)
