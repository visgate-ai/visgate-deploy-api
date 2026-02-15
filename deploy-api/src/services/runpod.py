"""Runpod Provider Implementation."""

from typing import Any, Optional

import httpx

from src.core.errors import RunpodAPIError
from src.core.logging import structured_log
from src.core.telemetry import record_runpod_api_error, span
from src.services.base_provider import BaseInferenceProvider, ProviderEndpoint
from src.services.provider_factory import register_provider


class RunpodProvider(BaseInferenceProvider):
    def __init__(self, graphql_url: str = "https://api.runpod.io/graphql"):
        self.graphql_url = graphql_url

    async def _graphql_request(
        self,
        api_key: str,
        query: str,
        variables: Optional[dict[str, Any]] = None,
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
        gpu_id: str,
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
        
        input_obj = {
            "name": name,
            "templateId": template_id,
            "gpuIds": gpu_id,
            "idleTimeout": kwargs.get("idle_timeout", 300),
            "locations": kwargs.get("locations", "US"),
            "scalerType": kwargs.get("scaler_type", "QUEUE_DELAY"),
            "scalerValue": kwargs.get("scaler_value", 2),
            "workersMin": kwargs.get("workers_min", 1),
            "workersMax": kwargs.get("workers_max", 2),
            "env": runpod_env
        }
        
        if kwargs.get("volume_in_gb"):
            input_obj["volumeInGb"] = kwargs["volume_in_gb"]
            input_obj["volumeMountPath"] = kwargs.get("volume_mount_path", "/runpod-volume")

        with span("runpod.create_endpoint", {"name": name}):
            structured_log("INFO", f"Runpod create_endpoint payload: {input_obj}")
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

    def get_run_url(self, endpoint_id: str) -> str:
        return f"https://api.runpod.ai/v2/{endpoint_id}/run"

# Register the provider
register_provider("runpod", RunpodProvider())
