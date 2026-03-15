"""Vast.ai On-Demand Instance Provider Implementation.

Vast.ai on-demand instances provide direct GPU access with public IP:port
connectivity — no PyWorker or serverless routing required.

Flow:
  1. create_endpoint  → search GPU offers → PUT /api/v0/asks/<offer_id>/ →
                        instance_id (contract)
  2. health check     → GET /api/v0/instances/<id>/ → check actual_status →
                        extract public_ipaddr:port → probe /health directly
  3. submit_job       → POST /run on worker (direct HTTP via instance IP:port)
  4. get_job_status   → GET /status/<id> on worker
  5. delete_endpoint  → DELETE /api/v0/instances/<id>/

Auth: Bearer token in Authorization header.
Management API: https://cloud.vast.ai
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from typing import Any

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

_CONSOLE_BASE = "https://cloud.vast.ai"

# Port the HTTP inference server listens on inside the container.
_WORKER_HTTP_PORT = 8000

# Separator used to encode worker URL inside the provider job ID so that
# get_job_status can reach the exact worker that accepted the job.
_JOB_ID_SEP = "||"


def _normalize_gpu_name_for_search(gpu_name: str) -> str:
    normalized = (gpu_name or "").strip()
    for prefix in ("NVIDIA ", "GeForce "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized.replace(" ", "_")


def _build_search_params(gpu_ram: int, gpu_names: list[str] | None = None) -> str:
    """Legacy string format — kept for backward compat in unit tests."""
    parts = ["verified=true", "rentable=true", "rented=false"]
    if gpu_ram > 0:
        parts.append(f"gpu_ram>={gpu_ram}")
    normalized_gpu_names = [_normalize_gpu_name_for_search(name) for name in (gpu_names or []) if name]
    if normalized_gpu_names:
        unique_gpu_names = list(dict.fromkeys(normalized_gpu_names))
        if len(unique_gpu_names) == 1:
            parts.append(f"gpu_name={unique_gpu_names[0]}")
        else:
            joined_gpu_names = ", ".join(unique_gpu_names)
            parts.append(f"gpu_name in [{joined_gpu_names}]")
    return " ".join(parts)


def _build_query_dict(gpu_ram_gb: int = 0, gpu_names: list[str] | None = None) -> dict[str, Any]:
    """Build a JSON query dict for the bundles API.

    Format: {"field": {"op": value}, ...}
    gpu_ram_gb is converted to MB for the API.
    """
    q: dict[str, Any] = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
    }
    if gpu_ram_gb > 0:
        q["gpu_ram"] = {"gte": gpu_ram_gb * 1024}  # API stores MB
    # Normalize GPU names: strip vendor prefixes but keep spaces
    # (the JSON dict API uses spaces, unlike the CLI string format)
    normalized = []
    for n in (gpu_names or []):
        name = (n or "").strip()
        for prefix in ("NVIDIA ", "GeForce "):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name:
            normalized.append(name)
    unique = list(dict.fromkeys(normalized))
    if len(unique) == 1:
        q["gpu_name"] = {"eq": unique[0]}
    elif len(unique) > 1:
        q["gpu_name"] = {"in": unique}
    return q


class VastProvider(BaseInferenceProvider):
    """Vast.ai inference provider using on-demand GPU instances."""

    def __init__(self, console_base: str = _CONSOLE_BASE) -> None:
        self.console_base = console_base.rstrip("/")

    # ── low-level HTTP helper ────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        api_key: str,
        *,
        base: str | None = None,
        json_payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
        _max_retries: int = 3,
    ) -> Any:
        url = f"{base or self.console_base}{path}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        if json_payload is not None:
            headers["Content-Type"] = "application/json"

        last_exc: Exception | None = None
        for attempt in range(_max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.request(
                        method,
                        url,
                        json=json_payload,
                        headers=headers,
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

    # ── On-demand instance helpers ───────────────────────────────────────

    async def search_offers(
        self,
        api_key: str,
        *,
        gpu_ram_gb: int = 0,
        gpu_names: list[str] | None = None,
        order: str = "dph_total",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search marketplace for available GPU offers via /api/v0/bundles/."""
        q = _build_query_dict(gpu_ram_gb, gpu_names)
        qs = urllib.parse.urlencode({"q": json.dumps(q), "order": order, "limit": limit})
        path = f"/api/v0/bundles/?{qs}"
        data = await self._request("GET", path, api_key)
        offers = data if isinstance(data, list) else data.get("offers", [])
        return offers

    async def create_instance(
        self,
        api_key: str,
        offer_id: int,
        *,
        image: str,
        env: dict[str, str],
        disk: int = 20,
        label: str = "",
    ) -> dict[str, Any]:
        """PUT /api/v0/asks/<offer_id>/ → {success, new_contract}."""
        # Build env dict in Vast format: env vars + port mappings
        vast_env: dict[str, str] = {}
        for k, v in env.items():
            vast_env[k] = v
        vast_env[f"-p {_WORKER_HTTP_PORT}:{_WORKER_HTTP_PORT}"] = "1"

        body: dict[str, Any] = {
            "image": image,
            "disk": disk,
            "runtype": "args",
            "env": vast_env,
        }
        if label:
            body["label"] = label
        return await self._request("PUT", f"/api/v0/asks/{offer_id}/", api_key, json_payload=body)

    async def get_instance(self, api_key: str, instance_id: int | str) -> dict[str, Any]:
        """GET /api/v0/instances/<id>/ → instance details."""
        return await self._request("GET", f"/api/v0/instances/{instance_id}/", api_key)

    async def destroy_instance(self, api_key: str, instance_id: int | str) -> Any:
        """DELETE /api/v0/instances/<id>/ → destroy on-demand instance."""
        return await self._request("DELETE", f"/api/v0/instances/{instance_id}/", api_key)

    # ── URL helpers ──────────────────────────────────────────────────────

    @staticmethod
    def build_endpoint_url(instance_id: str) -> str:
        """Encode instance ID as a virtual URL for storage."""
        return f"vast-inst://{instance_id}"

    @staticmethod
    def parse_endpoint_id(endpoint_url: str) -> str:
        """Extract instance ID from the virtual URL."""
        for prefix in ("vast-inst://", "vast-ep://"):
            if endpoint_url.startswith(prefix):
                rest = endpoint_url[len(prefix):]
                return rest.split("/", 1)[0]
        return endpoint_url

    @staticmethod
    def encode_job_id(worker_url: str, actual_job_id: str) -> str:
        return f"{worker_url}{_JOB_ID_SEP}{actual_job_id}"

    @staticmethod
    def decode_job_id(composite_id: str) -> tuple[str, str]:
        """Return (worker_url, actual_job_id)."""
        if _JOB_ID_SEP in composite_id:
            worker_url, actual = composite_id.split(_JOB_ID_SEP, 1)
            return worker_url, actual
        return "", composite_id

    @staticmethod
    def extract_worker_url(instance: dict[str, Any]) -> str | None:
        """Extract the direct HTTP URL from a Vast instance dict."""
        ip = instance.get("public_ipaddr")
        if not ip:
            return None
        ports = instance.get("ports", {})
        tcp_key = f"{_WORKER_HTTP_PORT}/tcp"
        tcp_mapping = ports.get(tcp_key, [])
        if isinstance(tcp_mapping, list) and tcp_mapping:
            host_port = tcp_mapping[0].get("HostPort")
            if host_port:
                return f"http://{ip}:{host_port}"
        return None

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
        """Create a Vast.ai on-demand GPU instance.

        ``gpu_id`` is interpreted as VRAM in GB (numeric string) or GPU name.
        """
        min_gpu_ram_gb = 0
        try:
            min_gpu_ram_gb = int(gpu_id)
        except (ValueError, TypeError):
            pass

        # Inject HTTP mode so the worker starts its HTTP server
        env = {**env, "WORKER_MODE": "http", "HTTP_PORT": str(_WORKER_HTTP_PORT)}

        candidate_gpu_ids = kwargs.get("gpu_ids")
        candidate_gpu_names = candidate_gpu_ids if isinstance(candidate_gpu_ids, list) else []
        if not candidate_gpu_names and gpu_id and not min_gpu_ram_gb:
            candidate_gpu_names = [gpu_id]

        structured_log(
            "INFO",
            "Vast.ai searching for GPU offers",
            metadata={
                "name": name,
                "image": image,
                "gpu_ram_gb": min_gpu_ram_gb,
                "candidate_gpu_names": candidate_gpu_names,
            },
        )

        # Search for matching GPU offers
        offers = await self.search_offers(
            api_key,
            gpu_ram_gb=min_gpu_ram_gb,
            gpu_names=candidate_gpu_names,
        )
        if not offers:
            raise VastAPIError("No matching GPU offers found on Vast.ai marketplace")

        # Pick cheapest offer
        offer = min(offers, key=lambda o: o.get("dph_total", 999))
        offer_id = offer["id"]
        offer_gpu = offer.get("gpu_name", "unknown")
        offer_price = offer.get("dph_total")

        structured_log(
            "INFO",
            "Vast.ai selected offer",
            metadata={
                "offer_id": offer_id,
                "gpu": offer_gpu,
                "price_per_hour": offer_price,
                "total_offers": len(offers),
            },
        )

        # Create on-demand instance
        inst_resp = await self.create_instance(
            api_key,
            offer_id,
            image=image,
            env=env,
            label=f"visgate-{name}",
        )
        instance_id = inst_resp.get("new_contract")
        if not instance_id:
            raise VastAPIError(
                "Instance creation returned no contract ID",
                details={"response": inst_resp},
            )

        structured_log(
            "INFO",
            "Vast.ai instance created",
            metadata={"instance_id": instance_id, "offer_id": offer_id, "gpu": offer_gpu},
        )

        return {
            "id": str(instance_id),
            "url": self.build_endpoint_url(str(instance_id)),
            "raw_response": {
                "instance_id": instance_id,
                "offer_id": offer_id,
                "gpu_name": offer_gpu,
                "price_per_hour": offer_price,
            },
        }

    async def delete_endpoint(self, endpoint_id: str, api_key: str) -> None:
        """Destroy the on-demand instance."""
        await self.destroy_instance(api_key, endpoint_id)

    async def list_endpoints(self, api_key: str) -> list[ProviderEndpointSummary]:
        resp = await self._request("GET", "/api/v0/instances/", api_key)
        instances = resp if isinstance(resp, list) else resp.get("instances", [])
        summaries: list[ProviderEndpointSummary] = []
        for inst in instances:
            inst_id = str(inst.get("id", ""))
            label = inst.get("label", "")
            status = inst.get("actual_status", "unknown")
            summaries.append({
                "id": inst_id,
                "name": label or f"vast-{inst_id}",
                "status": status,
                "url": self.build_endpoint_url(inst_id) if inst_id else None,
                "raw_response": inst,
            })
        return summaries

    async def list_gpu_types(self, api_key: str) -> list[dict[str, Any]]:
        """List available GPUs by querying the offers marketplace."""
        data = await self._request("GET", "/api/v0/bundles/", api_key)
        offers = data if isinstance(data, list) else data.get("offers", [])
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
        return self.build_endpoint_url(endpoint_id)

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
        """Submit a job directly to the worker's HTTP server."""
        # Resolve worker URL
        if endpoint_url.startswith("http://") or endpoint_url.startswith("https://"):
            worker_url = endpoint_url.rstrip("/")
        else:
            # Virtual URL — look up instance to get IP:port
            instance_id = self.parse_endpoint_id(endpoint_url)
            instance = await self.get_instance(api_key, instance_id)
            worker_url = self.extract_worker_url(instance)
            if not worker_url:
                raise VastAPIError(
                    "Instance has no public IP:port yet",
                    details={"instance_id": instance_id, "status": instance.get("actual_status")},
                )

        # Build job payload
        payload: dict[str, Any] = {"input": dict(job_input)}
        if s3_config:
            payload["input"]["s3Config"] = s3_config

        run_url = f"{worker_url}/run"
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
        actual_job_id = data.get("id", "")

        return {
            "id": self.encode_job_id(worker_url, actual_job_id),
            "status": data.get("status", "IN_QUEUE"),
            "raw_response": data,
        }

    async def get_job_status(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        worker_url, actual_job_id = self.decode_job_id(job_id)
        if not worker_url:
            raise VastAPIError(
                "Cannot determine worker URL from job ID",
                details={"job_id": job_id},
            )

        status_url = f"{worker_url.rstrip('/')}/status/{actual_job_id}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(status_url)
        if resp.status_code >= 400:
            raise VastAPIError(
                f"get_job_status HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        return {
            "id": job_id,
            "status": data.get("status", "UNKNOWN"),
            "output": data.get("output"),
            "error": data.get("error"),
            "delay_time": None,
            "execution_time": None,
            "cost_usd": None,
            "raw_response": data,
        }

    async def cancel_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        return await self.get_job_status(endpoint_url, job_id, api_key)

    async def retry_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        return await self.get_job_status(endpoint_url, job_id, api_key)

    async def get_endpoint_health(self, endpoint_url: str, api_key: str) -> dict[str, Any]:
        """Probe the instance's /health endpoint directly."""
        if endpoint_url.startswith(("http://", "https://")):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{endpoint_url.rstrip('/')}/health")
                if resp.status_code >= 400:
                    return {"status": "error", "http_status": resp.status_code}
                return resp.json()
            except Exception as exc:
                return {"status": "unreachable", "error": str(exc)}
        return {"status": "unknown", "note": "use check_endpoint_health with instance_id"}

    async def check_endpoint_health(self, endpoint_id: str, api_key: str, **kwargs: Any) -> dict[str, Any]:
        """Check instance status and probe the worker's /health endpoint directly."""
        try:
            instance = await self.get_instance(api_key, endpoint_id)
        except VastAPIError as exc:
            return {"status": "error", "error": str(exc), "workers": []}

        if isinstance(instance, str):
            return {"status": "error", "error": instance, "workers": []}

        if isinstance(instance, dict) and "error_msg" in instance:
            return {"status": "error", "error": instance["error_msg"], "workers": []}

        vast_status = instance.get("actual_status", "unknown")
        worker_url = self.extract_worker_url(instance)

        # Instance not running yet
        if vast_status not in ("running",):
            return {
                "status": "loading" if vast_status in ("loading", "creating", "pulling") else vast_status,
                "workers": [{"status": vast_status}],
                "running_count": 0,
                "total_count": 1,
                "vast_status": vast_status,
            }

        # Instance running but no public IP/port yet
        if not worker_url:
            return {
                "status": "loading",
                "workers": [{"status": "running_no_port"}],
                "running_count": 0,
                "total_count": 1,
                "vast_status": vast_status,
            }

        # Instance running with IP:port — probe /health
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{worker_url}/health")
            if resp.status_code == 200:
                body = resp.json()
                w_ready = body.get("workers", {}).get("ready", 0)
                if w_ready > 0:
                    structured_log(
                        "INFO",
                        "Vast instance model ready",
                        metadata={"instance_id": endpoint_id, "worker_url": worker_url},
                    )
                    return {
                        "status": "ready",
                        "workers": [{"status": "ready"}],
                        "running_count": 1,
                        "total_count": 1,
                        "worker_url": worker_url,
                    }
                # Server is up but model not loaded yet
                return {
                    "status": "loading",
                    "workers": [{"status": "model_loading"}],
                    "running_count": 0,
                    "total_count": 1,
                    "worker_url": worker_url,
                    "health_body": body,
                }
        except Exception as exc:
            # Server not responding yet (container starting)
            return {
                "status": "loading",
                "workers": [{"status": "server_starting"}],
                "running_count": 0,
                "total_count": 1,
                "vast_status": vast_status,
                "worker_url": worker_url,
                "probe_error": str(exc),
            }


# Register the provider
register_provider("vast", VastProvider())
