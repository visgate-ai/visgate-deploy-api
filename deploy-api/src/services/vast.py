"""Vast.ai Provider Implementation.

Vast.ai is an instance-based GPU marketplace.  Unlike RunPod's built-in
serverless queue, Vast.ai instances are raw Docker containers with exposed
ports.  The inference worker must run in HTTP mode (WORKER_MODE=http) so
that submit_job / get_job_status can communicate over plain HTTP.

Flow:
  1. create_endpoint  → search GPU offers → rent cheapest instance → wait "running"
  2. submit_job       → POST /run   to instance HTTP server → returns job ID
  3. get_job_status   → GET /status/<id> from instance → returns status/output
  4. delete_endpoint  → destroy instance

Auth: every request carries ``api_key`` as a query parameter.
"""

from __future__ import annotations

import asyncio
import json as _json
from typing import Any
from urllib.parse import quote

import httpx

from src.core.errors import VastAPIError
from src.core.logging import structured_log
from src.services.base_provider import (
    BaseInferenceProvider,
    ProviderEndpoint,
    ProviderEndpointSummary,
    ProviderJobAccepted,
    ProviderJobStatus,
)
from src.services.provider_factory import register_provider

_API_BASE = "https://cloud.vast.ai/api/v0"

# Port the HTTP inference server listens on inside the container.
_WORKER_HTTP_PORT = 8000

# How long to wait for an instance to reach "running" state.
_INSTANCE_READY_TIMEOUT = 600  # seconds
_INSTANCE_POLL_INTERVAL = 10   # seconds


class VastProvider(BaseInferenceProvider):
    """Vast.ai inference provider backed by on-demand GPU instances."""

    def __init__(self, api_base: str = _API_BASE) -> None:
        self.api_base = api_base.rstrip("/")

    # ── low-level HTTP helpers ───────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        api_key: str,
        *,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        timeout: float = 30.0,
        _max_retries: int = 3,
    ) -> Any:
        url = f"{self.api_base}{path}"
        query = {"api_key": api_key}
        if params:
            query.update(params)

        last_exc: Exception | None = None
        for attempt in range(_max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.request(
                        method,
                        url,
                        json=json_payload,
                        params=query,
                        headers={"Accept": "application/json"},
                    )

                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = VastAPIError(
                        message=f"HTTP {resp.status_code}: {resp.text[:500]}",
                        details={"status": resp.status_code},
                    )
                    if attempt < _max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise last_exc

                if resp.status_code >= 400:
                    raise VastAPIError(
                        message=f"HTTP {resp.status_code}: {resp.text[:500]}",
                        status_code=resp.status_code,
                        details={"status": resp.status_code},
                    )

                # Vast sometimes returns empty body on success (e.g. DELETE)
                text = resp.text.strip()
                if not text:
                    return {}
                return resp.json()

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = VastAPIError(message=f"Network error: {exc}")
                if attempt < _max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise last_exc from exc

        raise last_exc or VastAPIError(message="Vast.ai request failed after retries")

    # ── offer search ─────────────────────────────────────────────────────

    async def search_offers(
        self,
        api_key: str,
        *,
        min_gpu_ram_gb: int = 0,
        gpu_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search available on-demand GPU offers on Vast.ai.

        Returns a list of offers sorted by price (cheapest first).
        """
        query: dict[str, Any] = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
        }
        if min_gpu_ram_gb > 0:
            query["gpu_ram"] = {"gte": min_gpu_ram_gb * 1024}  # Vast uses MB
        if gpu_name:
            query["gpu_name"] = {"eq": gpu_name}

        q_json = _json.dumps(query)
        params = {
            "q": q_json,
            "order": "[[\"dph_total\",\"asc\"]]",
            "type": "on-demand",
            "limit": str(limit),
        }
        data = await self._request("GET", "/bundles/", api_key, params=params)
        offers = data if isinstance(data, list) else data.get("offers", [])
        return offers

    # ── instance lifecycle ───────────────────────────────────────────────

    async def _create_instance(
        self,
        api_key: str,
        offer_id: int,
        image: str,
        env: dict[str, str],
        disk_gb: float = 50.0,
        label: str = "",
    ) -> dict[str, Any]:
        """Rent a machine from an offer and spin up a Docker container."""
        body: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "env": env,
            "disk": disk_gb,
            "runtype": "args",
        }
        if label:
            body["label"] = label
        # PUT /asks/{offer_id}/ creates the instance
        return await self._request("PUT", f"/asks/{offer_id}/", api_key, json_payload=body)

    async def _get_instance(self, api_key: str, instance_id: int | str) -> dict[str, Any]:
        return await self._request("GET", f"/instances/{instance_id}/", api_key)

    async def _list_instances(self, api_key: str) -> list[dict[str, Any]]:
        data = await self._request("GET", "/instances/", api_key)
        if isinstance(data, list):
            return data
        return data.get("instances", [])

    async def _destroy_instance(self, api_key: str, instance_id: int | str) -> None:
        await self._request("DELETE", f"/instances/{instance_id}/", api_key)

    def _instance_endpoint_url(self, instance: dict[str, Any]) -> str | None:
        """Extract HTTP endpoint URL from a running Vast.ai instance."""
        public_ip = instance.get("public_ipaddr")
        if not public_ip:
            return None

        # Vast.ai provides direct port mapping: the instance exposes ports
        # accessible via public_ip:ssh_port offset or direct_port_start.
        # For HTTP server on port 8000, Vast.ai maps it to a host port.
        ports = instance.get("ports", {})
        port_key = f"{_WORKER_HTTP_PORT}/tcp"
        if port_key in ports:
            mappings = ports[port_key]
            if isinstance(mappings, list) and mappings:
                host_port = mappings[0].get("HostPort")
                if host_port:
                    return f"http://{public_ip}:{host_port}"

        # Fallback: Vast.ai often uses direct_port_start + offset
        direct_port_start = instance.get("direct_port_start")
        if direct_port_start:
            # Direct port mapping: container port 8000 → host direct_port_start + (8000 - first_exposed_port)
            # By default, if only 8000 is exposed, the mapping is direct_port_start itself
            return f"http://{public_ip}:{direct_port_start}"

        return None

    async def _wait_for_running(
        self,
        api_key: str,
        instance_id: int | str,
        timeout: float = _INSTANCE_READY_TIMEOUT,
    ) -> dict[str, Any]:
        """Poll instance until it reaches 'running' status or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            instance = await self._get_instance(api_key, instance_id)
            status = instance.get("actual_status", "")
            if status == "running":
                return instance
            if status in ("exited", "error", "destroyed"):
                raise VastAPIError(
                    f"Instance {instance_id} entered terminal state: {status}",
                    details={"instance_id": instance_id, "status": status},
                )
            if asyncio.get_event_loop().time() > deadline:
                raise VastAPIError(
                    f"Instance {instance_id} did not start within {timeout}s (status: {status})",
                    details={"instance_id": instance_id, "status": status},
                )
            await asyncio.sleep(_INSTANCE_POLL_INTERVAL)

    # ── BaseInferenceProvider implementation ─────────────────────────────

    async def create_endpoint(
        self,
        name: str,
        gpu_id: str,
        image: str,
        env: dict[str, str],
        api_key: str,
        **kwargs: Any,
    ) -> ProviderEndpoint:
        """Search for a GPU offer and create an instance.

        ``gpu_id`` is interpreted as a minimum VRAM requirement in GB (string).
        If it's a numeric string → search by VRAM; otherwise treat as GPU name.
        """
        min_gpu_ram_gb = 0
        gpu_name: str | None = None
        try:
            min_gpu_ram_gb = int(gpu_id)
        except (ValueError, TypeError):
            gpu_name = gpu_id

        # Inject HTTP mode so the worker starts its HTTP server instead of runpod.serverless
        env = {**env, "WORKER_MODE": "http", "HTTP_PORT": str(_WORKER_HTTP_PORT)}

        disk_gb = kwargs.get("disk_gb", 50.0)

        structured_log(
            "INFO",
            "Vast.ai searching for GPU offers",
            metadata={"min_gpu_ram_gb": min_gpu_ram_gb, "gpu_name": gpu_name, "image": image},
        )
        offers = await self.search_offers(
            api_key,
            min_gpu_ram_gb=min_gpu_ram_gb,
            gpu_name=gpu_name,
        )
        if not offers:
            raise VastAPIError(
                f"No Vast.ai offers found for GPU requirement: {gpu_id}",
                details={"gpu_id": gpu_id},
            )

        # Pick the cheapest suitable offer
        offer = offers[0]
        offer_id = offer["id"]
        structured_log(
            "INFO",
            "Vast.ai selected offer",
            metadata={
                "offer_id": offer_id,
                "gpu_name": offer.get("gpu_name"),
                "gpu_ram": offer.get("gpu_ram"),
                "dph_total": offer.get("dph_total"),
            },
        )

        result = await self._create_instance(
            api_key,
            offer_id,
            image,
            env,
            disk_gb=disk_gb,
            label=name,
        )

        instance_id = result.get("new_contract") or result.get("instance_id") or result.get("id")
        if not instance_id:
            raise VastAPIError("create_instance returned no instance ID", details={"response": result})

        structured_log("INFO", "Vast.ai instance created, waiting for running state", metadata={"instance_id": instance_id})
        instance = await self._wait_for_running(api_key, instance_id)
        endpoint_url = self._instance_endpoint_url(instance)
        if not endpoint_url:
            raise VastAPIError(
                f"Could not determine endpoint URL for instance {instance_id}",
                details={"instance": instance},
            )

        structured_log("INFO", "Vast.ai instance running", metadata={"instance_id": instance_id, "endpoint_url": endpoint_url})
        return {
            "id": str(instance_id),
            "url": endpoint_url,
            "raw_response": instance,
        }

    async def delete_endpoint(self, endpoint_id: str, api_key: str) -> None:
        await self._destroy_instance(api_key, endpoint_id)

    async def list_endpoints(self, api_key: str) -> list[ProviderEndpointSummary]:
        instances = await self._list_instances(api_key)
        summaries: list[ProviderEndpointSummary] = []
        for inst in instances:
            inst_id = str(inst.get("id", ""))
            status = inst.get("actual_status", "unknown")
            url = self._instance_endpoint_url(inst)
            summaries.append({
                "id": inst_id,
                "name": inst.get("label", "") or f"vast-{inst_id}",
                "status": status,
                "url": url,
                "raw_response": inst,
            })
        return summaries

    async def list_gpu_types(self, api_key: str) -> list[dict[str, Any]]:
        offers = await self.search_offers(api_key, limit=100)
        seen: dict[str, dict[str, Any]] = {}
        for o in offers:
            gpu = o.get("gpu_name", "unknown")
            if gpu not in seen:
                seen[gpu] = {
                    "id": gpu,
                    "displayName": gpu,
                    "memoryInGb": round((o.get("gpu_ram") or 0) / 1024, 1),
                    "pricePerHour": o.get("dph_total"),
                }
        return list(seen.values())

    def get_run_url(self, endpoint_id: str) -> str:
        # For Vast.ai the actual URL is determined per-instance and stored
        # in the deployment doc.  This method returns a placeholder.
        return f"vast://{endpoint_id}/run"

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
        """Submit an async inference job to the instance's HTTP server."""
        payload: dict[str, Any] = {"input": dict(job_input)}
        if s3_config:
            payload["input"]["s3Config"] = s3_config

        # endpoint_url is like http://1.2.3.4:54321
        run_url = endpoint_url.rstrip("/") + "/run"
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                run_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code >= 400:
            raise VastAPIError(
                f"submit_job HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        return {
            "id": data.get("id", ""),
            "status": data.get("status", "IN_QUEUE"),
            "raw_response": data,
        }

    async def get_job_status(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        status_url = endpoint_url.rstrip("/") + f"/status/{job_id}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(status_url)
        if resp.status_code >= 400:
            raise VastAPIError(
                f"get_job_status HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        return {
            "id": data.get("id", job_id),
            "status": data.get("status", "UNKNOWN"),
            "output": data.get("output"),
            "error": data.get("error"),
            "delay_time": None,
            "execution_time": None,
            "cost_usd": None,
            "raw_response": data,
        }

    async def cancel_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        # Vast.ai HTTP mode doesn't support cancel; return current status
        return await self.get_job_status(endpoint_url, job_id, api_key)

    async def retry_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        # Vast.ai HTTP mode doesn't support retry natively; return current status
        return await self.get_job_status(endpoint_url, job_id, api_key)

    async def get_endpoint_health(self, endpoint_url: str, api_key: str) -> dict[str, Any]:
        """Probe the instance HTTP server's /health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{endpoint_url.rstrip('/')}/health")
            if resp.status_code >= 400:
                return {"status": "error", "http_status": resp.status_code}
            return resp.json()
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}

    async def check_endpoint_health(self, endpoint_id: str, api_key: str) -> dict[str, Any]:
        """Check health by looking up the instance and probing its HTTP server."""
        instance = await self._get_instance(api_key, endpoint_id)
        url = self._instance_endpoint_url(instance)
        if not url:
            return {"status": "no_url", "actual_status": instance.get("actual_status")}
        return await self.get_endpoint_health(url, api_key)


# Register the provider
register_provider("vast", VastProvider())
