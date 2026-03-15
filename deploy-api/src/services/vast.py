"""Vast.ai Serverless Provider Implementation.

Vast.ai Serverless provides managed inference endpoints with auto-scaling
worker groups — similar to RunPod's serverless product.

Flow:
  1. create_endpoint  → create Vast template → create serverless endpoint →
                        create workergroup (links template + GPU requirements)
  2. submit_job       → POST /route/ (run.vast.ai) to get a live worker URL →
                        POST /run on that worker → return composite job ID
  3. get_job_status   → parse worker URL from composite job ID →
                        GET /status/<id> on that worker
  4. delete_endpoint  → DELETE /api/v0/endptjobs/<id>/
  5. health check     → POST /get_endpoint_workers/ (run.vast.ai)

Auth: Bearer token in Authorization header.
Management API: https://console.vast.ai
Routing API:    https://run.vast.ai
"""

from __future__ import annotations

import asyncio
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

_CONSOLE_BASE = "https://console.vast.ai"
_ROUTE_BASE = "https://run.vast.ai"

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


class VastProvider(BaseInferenceProvider):
    """Vast.ai inference provider using the Serverless API."""

    def __init__(
        self,
        console_base: str = _CONSOLE_BASE,
        route_base: str = _ROUTE_BASE,
    ) -> None:
        self.console_base = console_base.rstrip("/")
        self.route_base = route_base.rstrip("/")

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

    # ── Serverless management helpers ────────────────────────────────────

    async def create_template(
        self,
        api_key: str,
        *,
        name: str,
        image: str,
        env_str: str = "",
        runtype: str = "args",
    ) -> dict[str, Any]:
        """POST /api/v0/template/ → {success, template: {id, hash_id, name}}"""
        body: dict[str, Any] = {
            "name": name,
            "image": image,
            "runtype": runtype,
        }
        if env_str:
            body["env"] = env_str
        return await self._request("POST", "/api/v0/template/", api_key, json_payload=body)

    async def create_serverless_endpoint(
        self,
        api_key: str,
        *,
        endpoint_name: str,
        cold_workers: int = 0,
        max_workers: int = 1,
        min_load: int = 0,
        target_util: int = 80,
    ) -> dict[str, Any]:
        """POST /api/v0/endptjobs/ → {success, result: endpoint_id}"""
        body = {
            "endpoint_name": endpoint_name,
            "cold_workers": cold_workers,
            "max_workers": max_workers,
            "min_load": min_load,
            "target_util": target_util,
        }
        return await self._request("POST", "/api/v0/endptjobs/", api_key, json_payload=body)

    async def create_workergroup(
        self,
        api_key: str,
        *,
        endpoint_name: str,
        template_hash: str,
        gpu_ram: int = 0,
        cold_workers: int = 0,
        max_workers: int = 1,
        gpu_names: list[str] | None = None,
        search_params: str | None = None,
    ) -> dict[str, Any]:
        """POST /api/v0/workergroups/ → {success, id}"""
        # search_params is required by Vast API and must be a string
        if not search_params:
            search_params = _build_search_params(gpu_ram, gpu_names)

        body: dict[str, Any] = {
            "endpoint_name": endpoint_name,
            "template_hash": template_hash,
            "search_params": search_params,
            "cold_workers": cold_workers,
            "max_workers": max_workers,
        }
        if gpu_ram > 0:
            body["gpu_ram"] = gpu_ram
        return await self._request("POST", "/api/v0/workergroups/", api_key, json_payload=body)

    async def route_request(
        self,
        api_key: str,
        endpoint_name: str,
        cost: int = 100,
    ) -> dict[str, Any]:
        """POST /route/ on run.vast.ai → {url, reqnum, signature} or {status: "Stopped"}"""
        body = {"endpoint": endpoint_name, "cost": cost}
        return await self._request(
            "POST", "/route/", api_key,
            base=self.route_base,
            json_payload=body,
        )

    async def get_endpoint_workers(
        self,
        api_key: str,
        endpoint_id: int | str,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """POST /get_endpoint_workers/ on run.vast.ai → list or {workers: [...]}"""
        body = {"id": int(endpoint_id)}
        return await self._request(
            "POST", "/get_endpoint_workers/", api_key,
            base=self.route_base,
            json_payload=body,
        )

    # ── URL helpers ──────────────────────────────────────────────────────

    @staticmethod
    def build_endpoint_url(endpoint_id: str) -> str:
        """Encode endpoint ID as a virtual URL for storage."""
        return f"vast-ep://{endpoint_id}"

    @staticmethod
    def parse_endpoint_id(endpoint_url: str) -> str:
        """Extract endpoint ID from the virtual URL."""
        prefix = "vast-ep://"
        if endpoint_url.startswith(prefix):
            return endpoint_url[len(prefix):]
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
    def _env_dict_to_flag_str(env: dict[str, str]) -> str:
        """Convert env dict to Docker-flag format for Vast template API.

        Example: ``{A: 1, B: 2}`` → ``-e A=1 -e B=2 -p 8000:8000``
        """
        parts = [f"-e {k}={v}" for k, v in env.items()]
        parts.append(f"-p {_WORKER_HTTP_PORT}:{_WORKER_HTTP_PORT}")
        return " ".join(parts)

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
        """Create a Vast.ai serverless endpoint with template and workergroup.

        ``gpu_id`` is interpreted as VRAM in GB (numeric string) or GPU name.
        """
        min_gpu_ram_gb = 0
        try:
            min_gpu_ram_gb = int(gpu_id)
        except (ValueError, TypeError):
            pass

        # Inject HTTP mode so the worker starts its HTTP server
        env = {**env, "WORKER_MODE": "http", "HTTP_PORT": str(_WORKER_HTTP_PORT)}
        env_str = self._env_dict_to_flag_str(env)

        cold_workers = max(1, int(kwargs.get("cold_workers") or 1))
        max_workers = max(1, int(kwargs.get("max_workers") or 1))
        candidate_gpu_ids = kwargs.get("gpu_ids")
        candidate_gpu_names = candidate_gpu_ids if isinstance(candidate_gpu_ids, list) else []
        if not candidate_gpu_names and gpu_id and not min_gpu_ram_gb:
            candidate_gpu_names = [gpu_id]
        endpoint_name = name

        structured_log(
            "INFO",
            "Vast.ai creating serverless endpoint",
            metadata={
                "endpoint_name": endpoint_name,
                "image": image,
                "gpu_ram_gb": min_gpu_ram_gb,
                "candidate_gpu_names": candidate_gpu_names,
            },
        )

        # 1. Create template
        template_resp = await self.create_template(
            api_key,
            name=f"visgate-{endpoint_name}",
            image=image,
            env_str=env_str,
        )
        template = template_resp.get("template", {})
        template_hash = template.get("hash_id", "")
        template_id = template.get("id")
        if not template_hash:
            raise VastAPIError(
                "Template creation returned no hash_id",
                details={"response": template_resp},
            )
        structured_log("INFO", "Vast.ai template created", metadata={"template_id": template_id, "hash": template_hash})

        # 2. Create serverless endpoint
        ep_resp = await self.create_serverless_endpoint(
            api_key,
            endpoint_name=endpoint_name,
            cold_workers=cold_workers,
            max_workers=max_workers,
        )
        ep_id = ep_resp.get("result")
        if not ep_id:
            raise VastAPIError(
                "Endpoint creation returned no ID",
                details={"response": ep_resp},
            )
        structured_log("INFO", "Vast.ai serverless endpoint created", metadata={"endpoint_id": ep_id})

        # 3. Create workergroup connecting template → endpoint
        try:
            wg_resp = await self.create_workergroup(
                api_key,
                endpoint_name=endpoint_name,
                template_hash=template_hash,
                gpu_ram=min_gpu_ram_gb if min_gpu_ram_gb else 0,
                cold_workers=cold_workers,
                max_workers=max_workers,
                gpu_names=candidate_gpu_names,
            )
        except Exception:
            try:
                await self.delete_endpoint(str(ep_id), api_key)
                structured_log(
                    "WARNING",
                    "Vast.ai endpoint cleaned up after workergroup creation failure",
                    metadata={"endpoint_id": ep_id, "endpoint_name": endpoint_name},
                )
            except Exception as cleanup_exc:
                structured_log(
                    "WARNING",
                    "Vast.ai endpoint cleanup failed after workergroup creation failure",
                    metadata={
                        "endpoint_id": ep_id,
                        "endpoint_name": endpoint_name,
                        "cleanup_error": str(cleanup_exc),
                    },
                )
            raise
        wg_id = wg_resp.get("id")
        structured_log("INFO", "Vast.ai workergroup created", metadata={"workergroup_id": wg_id})

        return {
            "id": str(ep_id),
            "url": self.build_endpoint_url(str(ep_id)),
            "raw_response": {
                "endpoint_id": ep_id,
                "endpoint_name": endpoint_name,
                "template_id": template_id,
                "template_hash": template_hash,
                "workergroup_id": wg_id,
            },
        }

    async def delete_endpoint(self, endpoint_id: str, api_key: str) -> None:
        """DELETE /api/v0/endptjobs/<id>/ — also deletes associated workergroups."""
        await self._request("DELETE", f"/api/v0/endptjobs/{endpoint_id}/", api_key)

    async def list_endpoints(self, api_key: str) -> list[ProviderEndpointSummary]:
        resp = await self._request("GET", "/api/v0/endptjobs/", api_key)
        results = resp.get("results", []) if isinstance(resp, dict) else []
        summaries: list[ProviderEndpointSummary] = []
        for ep in results:
            ep_id = str(ep.get("id", ""))
            ep_name = ep.get("endpoint_name", "")
            summaries.append({
                "id": ep_id,
                "name": ep_name or f"vast-{ep_id}",
                "status": ep.get("endpoint_state", "unknown"),
                "url": self.build_endpoint_url(ep_id) if ep_id else None,
                "raw_response": ep,
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
        return f"{self.route_base}/route/"

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
        """Route to a live worker then POST /run on it.

        The worker URL is encoded in the returned job ID so that
        ``get_job_status`` can reach the correct worker later.
        """
        endpoint_id = self.parse_endpoint_id(endpoint_url)

        # Ask Vast router for a live worker
        route_resp = await self.route_request(api_key, endpoint_id)
        if isinstance(route_resp, dict) and route_resp.get("status") == "Stopped":
            raise VastAPIError(
                "No workers available (endpoint stopped)",
                details={"endpoint": endpoint_id},
            )
        worker_url = route_resp.get("url") if isinstance(route_resp, dict) else None
        if not worker_url:
            raise VastAPIError(
                "Route returned no worker URL",
                details={"endpoint": endpoint_id, "response": route_resp},
            )

        # Build job payload
        payload: dict[str, Any] = {"input": dict(job_input)}
        if s3_config:
            payload["input"]["s3Config"] = s3_config

        run_url = f"{worker_url.rstrip('/')}/run"
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
        """Check endpoint health via the worker-status API on run.vast.ai.

        ``endpoint_url`` is the stored virtual URL (``vast-ep://<id>``).
        We parse the endpoint ID and look up workers through the router.
        """
        # If this is a direct HTTP URL (e.g. from a worker), probe /health
        if endpoint_url.startswith(("http://", "https://")):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{endpoint_url.rstrip('/')}/health")
                if resp.status_code >= 400:
                    return {"status": "error", "http_status": resp.status_code}
                return resp.json()
            except Exception as exc:
                return {"status": "unreachable", "error": str(exc)}

        # For virtual endpoint URL, we cannot probe directly — caller should
        # use check_endpoint_health(endpoint_id, ...) instead.
        return {"status": "unknown", "note": "use check_endpoint_health with endpoint_id"}

    async def check_endpoint_health(self, endpoint_id: str, api_key: str) -> dict[str, Any]:
        """Query Vast.ai for the live workers of a serverless endpoint."""
        try:
            data = await self.get_endpoint_workers(api_key, endpoint_id)
        except VastAPIError as exc:
            return {"status": "error", "error": str(exc), "workers": []}

        if isinstance(data, str):
            return {"status": "error", "error": data, "workers": []}

        # Vast API may return {"error_msg": "..."} on auth or lookup failures
        if isinstance(data, dict) and "error_msg" in data:
            return {"status": "error", "error": data["error_msg"], "workers": []}

        # Vast.ai may return a list directly or a dict with a "workers" key
        if isinstance(data, list):
            workers = data
        else:
            workers = data.get("workers", [])
        # Vast.ai worker statuses: "creating", "loading", "ready", "idle", "stopped", "error"
        _READY_STATUSES = {"ready", "idle"}
        ready_workers = [w for w in workers if isinstance(w, dict) and w.get("status") in _READY_STATUSES]

        # Direct health probe: Vast PyWorker-based readiness may not apply to custom images.
        # If workers exist but none are Vast-ready, try probing the worker's HTTP /health directly.
        if not ready_workers and workers:
            probe_url = await self._find_probe_url(workers, endpoint_id, api_key)
            if probe_url:
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        resp = await client.get(f"{probe_url}/health")
                    if resp.status_code == 200:
                        body = resp.json()
                        w_ready = body.get("workers", {}).get("ready", 0)
                        if w_ready > 0:
                            structured_log(
                                "INFO",
                                "Direct health probe: worker model ready (bypassing Vast status)",
                                metadata={"endpoint_id": endpoint_id, "probe_url": probe_url},
                            )
                            return {
                                "status": "ready",
                                "workers": workers,
                                "running_count": 1,
                                "total_count": len(workers),
                                "direct_probe_url": probe_url,
                            }
                        else:
                            structured_log(
                                "DEBUG",
                                "Direct health probe: model not ready yet",
                                metadata={"endpoint_id": endpoint_id, "probe_url": probe_url, "body": body},
                            )
                except Exception as probe_exc:
                    structured_log(
                        "DEBUG",
                        "Direct health probe failed",
                        metadata={"endpoint_id": endpoint_id, "probe_url": probe_url, "error": str(probe_exc)},
                    )

        return {
            "status": "ready" if ready_workers else ("loading" if workers else "no_workers"),
            "workers": workers,
            "running_count": len(ready_workers),
            "total_count": len(workers),
        }

    @staticmethod
    def _extract_worker_url(worker: dict[str, Any]) -> str | None:
        """Try to extract a public HTTP URL from a Vast worker dict."""
        # Vast may include public_ipaddr + ports mapping
        ip = worker.get("public_ipaddr") or worker.get("ip_addr") or worker.get("addr")
        if ip:
            ports = worker.get("ports", {})
            # Look for port 8000/tcp mapping (our inference server port)
            tcp_8000 = ports.get("8000/tcp")
            if isinstance(tcp_8000, list) and tcp_8000:
                host_port = tcp_8000[0].get("HostPort")
                if host_port:
                    return f"http://{ip}:{host_port}"
            # Try direct port 8000 if ports not mapped
            if not ports:
                return f"http://{ip}:8000"
        return None

    async def _find_probe_url(
        self,
        workers: list[dict[str, Any]],
        endpoint_id: str,
        api_key: str,
    ) -> str | None:
        """Return a probe URL for the first loading worker.

        Strategy:
          1. Extract IP:port directly from the worker dict (cheapest).
          2. Fall back to calling the /route/ API which may return a URL
             even if the worker is still loading.
        """
        for w in workers:
            if not isinstance(w, dict):
                continue
            w_status = w.get("status", "")
            if w_status not in ("loading", "model_loading"):
                continue
            url = self._extract_worker_url(w)
            if url:
                return url

        # Fallback: ask the Vast router for a worker URL
        return await self._route_probe_url(endpoint_id, api_key)

    async def _route_probe_url(self, endpoint_id: str, api_key: str) -> str | None:
        """Call /route/ and return the worker URL if available."""
        try:
            route_resp = await self.route_request(api_key, endpoint_id)
            if isinstance(route_resp, dict) and route_resp.get("url"):
                return route_resp["url"].rstrip("/")
        except Exception:
            pass
        return None


# Register the provider
register_provider("vast", VastProvider())
