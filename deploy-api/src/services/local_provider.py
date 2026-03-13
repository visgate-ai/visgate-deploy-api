"""Local Docker/HTTP provider used for full local testing without RunPod."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx

from src.core.config import get_settings
from src.core.errors import RunpodAPIError
from src.services.base_provider import (
    BaseInferenceProvider,
    ProviderEndpoint,
    ProviderEndpointSummary,
    ProviderJobAccepted,
    ProviderJobStatus,
)
from src.services.gpu_registry import DEFAULT_GPU_REGISTRY
from src.services.provider_factory import register_provider


class LocalProvider(BaseInferenceProvider):
    """Talk to locally-running worker containers that expose a RunPod-like HTTP API."""

    def __init__(self) -> None:
        self._endpoints: dict[str, dict[str, Any]] = {}

    def _base_url_for_profile(self, profile: str | None) -> str:
        settings = get_settings()
        normalized = (profile or "image").strip().lower()
        if normalized == "audio":
            return settings.local_worker_url_audio.rstrip("/")
        if normalized == "video":
            return settings.local_worker_url_video.rstrip("/")
        return settings.local_worker_url_image.rstrip("/")

    def _endpoint_root(self, endpoint_url: str) -> str:
        return endpoint_url[:-4] if endpoint_url.endswith("/run") else endpoint_url.rstrip("/")

    def get_run_url(self, endpoint_id: str) -> str:
        endpoint = self._endpoints.get(endpoint_id)
        if not endpoint:
            raise RunpodAPIError(f"Unknown local endpoint: {endpoint_id}", status_code=404)
        return endpoint["url"]

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_payload: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                url,
                json=json_payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
        if response.status_code >= 400:
            raise RunpodAPIError(
                message=f"Local worker HTTP {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
            )
        if not response.text.strip():
            return {}
        return response.json()

    async def create_endpoint(
        self,
        name: str,
        gpu_ids: str | list[str],
        image: str,
        env: dict[str, str],
        api_key: str,
        **kwargs: Any,
    ) -> ProviderEndpoint:
        profile = kwargs.get("worker_profile") or env.get("WORKER_PROFILE") or "image"
        base_url = self._base_url_for_profile(profile)
        endpoint_id = f"local-{profile}-{uuid4().hex[:12]}"
        payload = {
            "endpoint_id": endpoint_id,
            "name": name,
            "profile": profile,
            "image": image,
            "gpu_ids": gpu_ids,
            "env": env,
        }
        data = await self._request("POST", f"{base_url}/load", json_payload=payload, timeout=30.0)
        endpoint_url = f"{base_url}/run"
        self._endpoints[endpoint_id] = {
            "id": endpoint_id,
            "name": name,
            "status": data.get("status", "LOADING"),
            "base_url": base_url,
            "url": endpoint_url,
            "profile": profile,
            "image": image,
        }
        return {"id": endpoint_id, "url": endpoint_url, "raw_response": data}

    async def delete_endpoint(self, endpoint_id: str, api_key: str) -> None:
        endpoint = self._endpoints.get(endpoint_id)
        if not endpoint:
            return
        try:
            await self._request("POST", f"{endpoint['base_url']}/reset", json_payload={"endpoint_id": endpoint_id}, timeout=15.0)
        finally:
            self._endpoints.pop(endpoint_id, None)

    async def delete_template(self, template_name: str, api_key: str) -> None:
        return None

    async def list_endpoints(self, api_key: str) -> list[ProviderEndpointSummary]:
        items: list[ProviderEndpointSummary] = []
        for endpoint_id, meta in list(self._endpoints.items()):
            status = meta.get("status", "UNKNOWN")
            try:
                health = await self.get_endpoint_health(meta["url"], api_key)
                if int((health.get("workers") or {}).get("ready", 0) or 0) > 0:
                    status = "READY"
                elif health.get("status") == "LOADING":
                    status = "LOADING"
            except Exception:
                pass
            items.append(
                {
                    "id": endpoint_id,
                    "name": meta.get("name", endpoint_id),
                    "status": status,
                    "url": meta.get("url"),
                    "raw_response": meta,
                }
            )
        return items

    async def list_gpu_types(self, api_key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for gpu in DEFAULT_GPU_REGISTRY:
            items.append(
                {
                    "id": gpu["id"],
                    "displayName": gpu["display"],
                    "memoryInGb": gpu["vram"],
                    "secureCloud": False,
                    "communityCloud": True,
                    "securePrice": None,
                    "communityPrice": 0.0,
                }
            )
        return items

    async def submit_job(
        self,
        endpoint_url: str,
        api_key: str,
        job_input: dict[str, Any],
        *,
        webhook_url: str | None = None,
        policy: dict[str, Any] | None = None,
        s3_config: dict[str, Any] | None = None,
    ) -> ProviderJobAccepted:
        payload_input = dict(job_input)
        if s3_config and "s3Config" not in payload_input:
            payload_input["s3Config"] = s3_config
        payload: dict[str, Any] = {"input": payload_input}
        if webhook_url:
            payload["webhook"] = webhook_url
        if policy:
            payload["policy"] = policy
        if s3_config:
            payload["s3Config"] = s3_config
        data = await self._request("POST", self._endpoint_root(endpoint_url) + "/run", json_payload=payload, timeout=30.0)
        return {"id": data.get("id", ""), "status": data.get("status", "IN_QUEUE"), "raw_response": data}

    async def get_job_status(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        data = await self._request("GET", f"{self._endpoint_root(endpoint_url)}/status/{job_id}", timeout=30.0)
        return {
            "id": data.get("id", job_id),
            "status": data.get("status", "UNKNOWN"),
            "output": data.get("output"),
            "error": data.get("error"),
            "delay_time": data.get("delayTime"),
            "execution_time": data.get("executionTime"),
            "cost_usd": data.get("costUSD") or data.get("costUsd") or data.get("cost"),
            "raw_response": data,
        }

    async def cancel_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        data = await self._request("POST", f"{self._endpoint_root(endpoint_url)}/cancel/{job_id}", timeout=30.0)
        return {"id": data.get("id", job_id), "status": data.get("status", "CANCELLED"), "raw_response": data}

    async def retry_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        data = await self._request("POST", f"{self._endpoint_root(endpoint_url)}/retry/{job_id}", timeout=30.0)
        return {"id": data.get("id", job_id), "status": data.get("status", "IN_QUEUE"), "raw_response": data}

    async def get_endpoint_health(self, endpoint_url: str, api_key: str) -> dict[str, Any]:
        return await self._request("GET", self._endpoint_root(endpoint_url) + "/health", timeout=15.0)

    async def check_endpoint_health(self, endpoint_id: str, api_key: str) -> dict[str, Any]:
        endpoint = self._endpoints.get(endpoint_id)
        if not endpoint:
            raise RunpodAPIError(f"Unknown local endpoint: {endpoint_id}", status_code=404)
        return await self._request("GET", f"{endpoint['base_url']}/health", timeout=15.0)


register_provider("local", LocalProvider())