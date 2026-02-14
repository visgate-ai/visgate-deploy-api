"""Runpod GraphQL API client for serverless endpoint CRUD."""

import asyncio
from typing import Any, Optional

import httpx

from src.core.errors import RunpodAPIError, RunpodInsufficientGPUError
from src.core.logging import structured_log
from src.core.telemetry import record_runpod_api_error, span
from src.services.gpu_registry import gpu_id_to_display_name, select_gpu_id_for_vram


RUNPOD_BASE = "https://api.runpod.io"


async def _graphql_request(
    api_key: str,
    query: str,
    variables: Optional[dict[str, Any]] = None,
    url: str = "https://api.runpod.io/graphql",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Send GraphQL request to Runpod."""
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            json=payload,
            params={"api_key": api_key},
            headers={"Content-Type": "application/json"},
        )
    if resp.status_code >= 400:
        body = resp.text
        try:
            data = resp.json()
            if "errors" in data and data["errors"]:
                record_runpod_api_error()
                raise RunpodAPIError(
                    message=data["errors"][0].get("message", body[:500]),
                    details={"errors": data["errors"], "status": resp.status_code},
                )
        except RunpodAPIError:
            raise
        except Exception:
            pass
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


# GraphQL: create endpoint (saveEndpoint)
MUTATION_SAVE_ENDPOINT = """
mutation SaveEndpoint($input: EndpointInput!) {
  saveEndpoint(input: $input) {
    id
    gpuIds
    name
    idleTimeout
    locations
    scalerType
    scalerValue
    templateId
    workersMax
    workersMin
  }
}
"""

# GraphQL: delete endpoint
MUTATION_DELETE_ENDPOINT = """
mutation DeleteEndpoint($id: String!) {
  deleteEndpoint(id: $id)
}
"""

# GraphQL: list endpoints (for status)
QUERY_MYSELF_ENDPOINTS = """
query Endpoints {
  myself {
    endpoints {
      id
      gpuIds
      name
      templateId
      workersMax
      workersMin
    }
  }
}
"""

# GraphQL: create serverless template (our inference image + optional default env)
MUTATION_SAVE_TEMPLATE = """
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

# GraphQL: list templates (to find existing or get id)
QUERY_MYSELF_TEMPLATES = """
query Templates {
  myself {
    templates {
      id
      name
      imageName
      isServerless
    }
  }
}
"""


def _env_dict_to_runpod_list(env: Optional[dict[str, str]]) -> list[dict[str, str]]:
    """Convert { K: V } to Runpod format [{ key: K, value: V }] (for saveTemplate)."""
    if not env:
        return []
    return [{"key": k, "value": str(v)} for k, v in env.items() if v is not None]


def _build_save_endpoint_input(
    name: str,
    template_id: str,
    gpu_ids: str,
    image: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    workers_min: int = 0,
    workers_max: int = 3,
    idle_timeout: int = 5,
    scaler_type: str = "QUEUE_DELAY",
    scaler_value: int = 4,
    locations: str = "US",
) -> dict[str, Any]:
    input_dict: dict[str, Any] = {
        "name": name,
        "templateId": template_id,
        "gpuIds": gpu_ids,
        "idleTimeout": idle_timeout,
        "locations": locations,
        "scalerType": scaler_type,
        "scalerValue": scaler_value,
        "workersMin": workers_min,
        "workersMax": workers_max,
        "networkVolumeId": "",
    }
    # Endpoint-level env: GraphQL EndpointInput may expect object { KEY: value }, not array
    if env:
        input_dict["env"] = {k: str(v) for k, v in env.items() if v is not None}
    return input_dict


async def create_endpoint(
    api_key: str,
    name: str,
    template_id: str,
    gpu_ids: str,
    *,
    env: Optional[dict[str, str]] = None,
    workers_min: int = 0,
    workers_max: int = 3,
    url: str = "https://api.runpod.io/graphql",
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Create a Runpod serverless endpoint via GraphQL saveEndpoint.
    Returns dict with id, gpuIds, etc. Endpoint run URL is https://api.runpod.ai/v2/{id}/run
    """
    with span("runpod.create_endpoint", {"name": name, "gpuIds": gpu_ids}):
        input_obj = _build_save_endpoint_input(
            name=name,
            template_id=template_id,
            gpu_ids=gpu_ids,
            env=env,
            workers_min=workers_min,
            workers_max=workers_max,
        )
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                data = await _graphql_request(
                    api_key=api_key,
                    query=MUTATION_SAVE_ENDPOINT,
                    variables={"input": input_obj},
                    url=url,
                )
                result = data.get("saveEndpoint")
                if not result:
                    raise RunpodAPIError(message="saveEndpoint returned no data")
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        record_runpod_api_error()
        raise RunpodAPIError(
            message=f"Failed after {max_retries} retries: {last_error}",
            details={"last_error": str(last_error)},
        )


async def delete_endpoint(
    api_key: str,
    endpoint_id: str,
    url: str = "https://api.runpod.io/graphql",
) -> None:
    """Delete a Runpod serverless endpoint. Workers must be 0 before delete."""
    with span("runpod.delete_endpoint", {"endpoint_id": endpoint_id}):
        await _graphql_request(
            api_key=api_key,
            query=MUTATION_DELETE_ENDPOINT,
            variables={"id": endpoint_id},
            url=url,
        )


def select_gpu(
    vram_gb: int,
    gpu_tier: Optional[str] = None,
) -> tuple[str, str]:
    """
    Select Runpod gpuId and display name for required VRAM.
    Returns (gpu_id, display_name). Raises RunpodInsufficientGPUError if none.
    """
    gpu_id = select_gpu_id_for_vram(vram_gb, gpu_tier)
    if not gpu_id:
        raise RunpodInsufficientGPUError(vram_gb)
    display = gpu_id_to_display_name(gpu_id)
    return gpu_id, display


def endpoint_run_url(endpoint_id: str) -> str:
    """Return the run URL for a serverless endpoint."""
    return f"https://api.runpod.ai/v2/{endpoint_id}/run"


async def create_serverless_template(
    api_key: str,
    name: str,
    image_name: str,
    *,
    container_disk_in_gb: int = 25,
    env: Optional[dict[str, str]] = None,
    url: str = "https://api.runpod.io/graphql",
) -> dict[str, Any]:
    """
    Create a Runpod serverless template with the given image.
    Use this template's id as RUNPOD_TEMPLATE_ID so endpoints run our inference image.
    Returns dict with id, name, imageName, etc.
    """
    # Serverless: volumeInGb 0; dockerArgs and env often required by API
    input_obj: dict[str, Any] = {
        "name": name,
        "imageName": image_name,
        "isServerless": True,
        "containerDiskInGb": container_disk_in_gb,
        "volumeInGb": 0,
        "dockerArgs": "",
        "env": _env_dict_to_runpod_list(env) if env else [],
    }
    with span("runpod.create_template", {"name": name, "imageName": image_name}):
        data = await _graphql_request(
            api_key=api_key,
            query=MUTATION_SAVE_TEMPLATE,
            variables={"input": input_obj},
            url=url,
        )
    result = data.get("saveTemplate")
    if not result:
        raise RunpodAPIError(message="saveTemplate returned no data")
    return result


async def list_templates(
    api_key: str,
    url: str = "https://api.runpod.io/graphql",
) -> list[dict[str, Any]]:
    """List Runpod templates for the current user."""
    data = await _graphql_request(
        api_key=api_key,
        query=QUERY_MYSELF_TEMPLATES,
        url=url,
    )
    myself = data.get("myself") or {}
    return myself.get("templates") or []
