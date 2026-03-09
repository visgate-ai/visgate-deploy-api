"""Runpod Provider Implementation."""

from typing import Any

import httpx

from src.core.errors import RunpodAPIError
from src.core.logging import structured_log
from src.core.telemetry import record_runpod_api_error, span
from src.services.base_provider import (
    BaseInferenceProvider,
    ProviderEndpoint,
    ProviderJobAccepted,
    ProviderJobStatus,
    ProviderEndpointSummary,
)
from src.services.provider_factory import register_provider


async def create_serverless_template(
    *,
    api_key: str,
    name: str,
    image_name: str,
    container_disk_in_gb: int = 20,
    volume_in_gb: int = 0,
    docker_args: str = "",
    readme: str = "",
    env: list[dict[str, str]] | None = None,
    template_id: str | None = None,
) -> dict[str, Any]:
    provider = RunpodProvider()
    mutation = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
      saveTemplate(input: $input) {
        id
        name
        imageName
        isServerless
        containerDiskInGb
        volumeInGb
      }
    }
    """
    input_obj: dict[str, Any] = {
        "name": name,
        "imageName": image_name,
        "isServerless": True,
        "containerDiskInGb": container_disk_in_gb,
        "volumeInGb": volume_in_gb,
        "dockerArgs": docker_args,
        "readme": readme,
        "env": env or [],
    }
    if template_id:
        input_obj["id"] = template_id
    data = await provider._graphql_request(api_key, mutation, {"input": input_obj})
    result = data.get("saveTemplate")
    if not result:
        raise RunpodAPIError("saveTemplate returned no data")
    return result


class RunpodProvider(BaseInferenceProvider):
    def __init__(self, graphql_url: str = "https://api.runpod.io/graphql"):
        self.graphql_url = graphql_url

    async def _graphql_request(
        self,
        api_key: str,
        query: str,
        variables: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.graphql_url,
                json=payload,
                params={"api_key": api_key},
                headers={"Content-Type": "application/json"},
            )

        if resp.status_code >= 400:
            body = resp.text
            record_runpod_api_error()
            raise RunpodAPIError(
                message=f"HTTP {resp.status_code}: {body[:500]}",
                details={"status": resp.status_code},
            )

        data = resp.json()
        if "errors" in data and data["errors"]:
            record_runpod_api_error()
            raise RunpodAPIError(
                message=data["errors"][0].get("message", "GraphQL error"),
                details={"errors": data["errors"]},
            )
        return data.get("data", {})

    async def create_endpoint(
        self,
        name: str,
        gpu_ids: list[str] | str,
        image: str,
        env: dict[str, str],
        api_key: str,
        **kwargs: Any
    ) -> ProviderEndpoint:
        template_id = kwargs.get("template_id")
        if not template_id:
             raise ValueError("Runpod requires template_id")

        mutation = """
        mutation SaveEndpoint($input: EndpointInput!) {
          saveEndpoint(input: $input) { id }
        }
        """


        runpod_env = [{"key": k, "value": str(v)} for k, v in env.items()]

        # Multi-GPU targeting support: join list into comma-separated string
        actual_gpu_ids = ",".join(gpu_ids) if isinstance(gpu_ids, list) else gpu_ids

        input_obj = {
            "name": name,
            "templateId": template_id,
            "gpuIds": actual_gpu_ids,
            "idleTimeout": kwargs.get("idle_timeout", 300),
            "executionTimeoutMs": kwargs.get("execution_timeout_ms", 300000),
            "locations": kwargs.get("locations", "US"),
            "scalerType": kwargs.get("scaler_type", "QUEUE_DELAY"),
            "scalerValue": kwargs.get("scaler_value", 2),
            "workersMin": kwargs.get("workers_min", 1),
            "workersMax": kwargs.get("workers_max", 1),
            "env": runpod_env
        }

        if kwargs.get("volume_in_gb"):
            input_obj["volumeInGb"] = kwargs["volume_in_gb"]
            input_obj["volumeMountPath"] = kwargs.get("volume_mount_path", "/runpod-volume")

        with span("runpod.create_endpoint", {"name": name}):
            structured_log("INFO", "Runpod create_endpoint requested", metadata={"name": name, "gpu_ids": actual_gpu_ids})
            data = await self._graphql_request(api_key, mutation, {"input": input_obj})
            result = data.get("saveEndpoint")
            if not result:
                raise RunpodAPIError("saveEndpoint returned no data")

            endpoint_id = result["id"]
            return {
                "id": endpoint_id,
                "url": self.get_run_url(endpoint_id),
                "raw_response": result
            }

    async def delete_endpoint(self, endpoint_id: str, api_key: str) -> None:
        mutation = """
        mutation DeleteEndpoint($id: String!) {
          deleteEndpoint(id: $id)
        }
        """
        await self._graphql_request(api_key, mutation, {"id": endpoint_id})

    async def delete_template(self, template_name: str, api_key: str) -> None:
        """Best-effort deletion of a per-deployment template.

        The template must not be in use by any running endpoint.  Call this
        only AFTER delete_endpoint has completed.  Failures are not propagated
        — the caller should catch and log them.
        """
        mutation = """
        mutation DeleteTemplate($name: String!) {
          deleteTemplate(templateName: $name)
        }
        """
        await self._graphql_request(api_key, mutation, {"name": template_name})

    async def list_endpoints(self, api_key: str) -> list[ProviderEndpointSummary]:
        query = """
        query MyEndpoints {
          myEndpoints {
            id
            name
            status
          }
        }
        """
        data = await self._graphql_request(api_key, query)
        endpoints = data.get("myEndpoints") or []
        summaries: list[ProviderEndpointSummary] = []
        for ep in endpoints:
            endpoint_id = ep.get("id")
            summaries.append(
                {
                    "id": endpoint_id,
                    "name": ep.get("name", ""),
                    "status": ep.get("status", ""),
                    "url": self.get_run_url(endpoint_id) if endpoint_id else None,
                    "raw_response": ep,
                }
            )
        return summaries

    async def list_gpu_types(self, api_key: str) -> list[dict[str, Any]]:
        query = """
        {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            securePrice
            communityPrice
          }
        }
        """
        data = await self._graphql_request(api_key, query)
        return data.get("gpuTypes") or []

    def get_run_url(self, endpoint_id: str) -> str:
        return f"https://api.runpod.ai/v2/{endpoint_id}/run"

    def _endpoint_root(self, endpoint_url: str) -> str:
        return endpoint_url[:-4] if endpoint_url.endswith("/run") else endpoint_url.rstrip("/")

    async def _endpoint_request(
        self,
        method: str,
        url: str,
        api_key: str,
        *,
        json_payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method,
                url,
                json=json_payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        if resp.status_code >= 400:
            raise RunpodAPIError(
                message=f"HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        return resp.json()

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
        data = await self._endpoint_request("POST", self._endpoint_root(endpoint_url) + "/run", api_key, json_payload=payload)
        return {"id": data.get("id", ""), "status": data.get("status", "IN_QUEUE"), "raw_response": data}

    async def get_job_status(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        data = await self._endpoint_request("GET", f"{self._endpoint_root(endpoint_url)}/status/{job_id}", api_key)
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
        data = await self._endpoint_request("POST", f"{self._endpoint_root(endpoint_url)}/cancel/{job_id}", api_key)
        return {"id": data.get("id", job_id), "status": data.get("status", "CANCELLED"), "raw_response": data}

    async def retry_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        data = await self._endpoint_request("POST", f"{self._endpoint_root(endpoint_url)}/retry/{job_id}", api_key)
        return {"id": data.get("id", job_id), "status": data.get("status", "IN_QUEUE"), "raw_response": data}

    async def get_endpoint_health(self, endpoint_url: str, api_key: str) -> dict[str, Any]:
        return await self._endpoint_request("GET", self._endpoint_root(endpoint_url) + "/health", api_key)

# Register the provider
register_provider("runpod", RunpodProvider())
