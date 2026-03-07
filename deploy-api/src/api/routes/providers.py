"""Provider key validation endpoints used by visgate-api BYOK flows."""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from src.core.config import get_settings
from src.models.schemas import ValidateKeyRequest, ValidateKeyResponse

router = APIRouter(prefix="/v1/providers", tags=["providers"])

SUPPORTED_PROVIDERS = ("fal", "replicate", "runway", "runpod", "huggingface")
_TIMEOUT_SECONDS = 10.0
_RUNWAY_VERSION = "2024-11-06"


async def _validate_fal_key(api_key: str) -> ValidateKeyResponse:
    auth_header = f"Key {api_key}" if ":" in api_key else f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        resp = await client.get(
            "https://rest.alpha.fal.ai/users/current",
            headers={"Authorization": auth_header},
        )
    if resp.status_code == 200:
        return ValidateKeyResponse(valid=True, message="Fal API key is valid")
    if resp.status_code in {401, 403}:
        return ValidateKeyResponse(valid=False, message="Fal API key is invalid")
    return ValidateKeyResponse(valid=False, message=f"Fal validation returned HTTP {resp.status_code}")


async def _validate_replicate_key(api_key: str) -> ValidateKeyResponse:
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        resp = await client.get(
            "https://api.replicate.com/v1/account",
            headers={"Authorization": f"Token {api_key}"},
        )
    if resp.status_code == 200:
        return ValidateKeyResponse(valid=True, message="Replicate API key is valid")
    if resp.status_code in {401, 403}:
        return ValidateKeyResponse(valid=False, message="Replicate API key is invalid")
    return ValidateKeyResponse(valid=False, message=f"Replicate validation returned HTTP {resp.status_code}")


async def _validate_runway_key(api_key: str) -> ValidateKeyResponse:
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        resp = await client.get(
            "https://api.dev.runwayml.com/v1/organization",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Runway-Version": _RUNWAY_VERSION,
            },
        )
    if resp.status_code == 200:
        return ValidateKeyResponse(valid=True, message="Runway API key is valid")
    if resp.status_code in {401, 403}:
        return ValidateKeyResponse(valid=False, message="Runway API key is invalid")
    return ValidateKeyResponse(valid=False, message=f"Runway validation returned HTTP {resp.status_code}")


async def _validate_runpod_key(api_key: str) -> ValidateKeyResponse:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            settings.runpod_graphql_url,
            json={"query": "{ myself { clientBalance } }"},
            params={"api_key": api_key},
            headers={"Content-Type": "application/json"},
        )
    if resp.status_code != 200:
        if resp.status_code in {401, 403}:
            return ValidateKeyResponse(valid=False, message="RunPod API key is invalid")
        return ValidateKeyResponse(valid=False, message=f"RunPod validation returned HTTP {resp.status_code}")

    payload = resp.json()
    if payload.get("errors"):
        return ValidateKeyResponse(
            valid=False,
            message=payload["errors"][0].get("message", "RunPod GraphQL validation failed"),
        )
    if payload.get("data", {}).get("myself") is None:
        return ValidateKeyResponse(valid=False, message="RunPod validation returned no user profile")
    return ValidateKeyResponse(valid=True, message="RunPod API key is valid")


async def _validate_huggingface_key(api_key: str) -> ValidateKeyResponse:
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        resp = await client.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if resp.status_code == 200:
        return ValidateKeyResponse(
            valid=True,
            message="Hugging Face token is valid",
        )
    if resp.status_code in {401, 403}:
        return ValidateKeyResponse(valid=False, message="Hugging Face token is invalid")
    return ValidateKeyResponse(valid=False, message=f"Hugging Face validation returned HTTP {resp.status_code}")


def _get_validator(provider: str):
    if provider == "fal":
        return _validate_fal_key
    if provider == "replicate":
        return _validate_replicate_key
    if provider == "runway":
        return _validate_runway_key
    if provider == "runpod":
        return _validate_runpod_key
    if provider == "huggingface":
        return _validate_huggingface_key
    return None


@router.post("/validate", response_model=ValidateKeyResponse, summary="Validate provider key")
async def validate_provider_key(request: ValidateKeyRequest) -> ValidateKeyResponse:
    """Validate a provider API key without storing it."""
    provider = request.provider.strip().lower()
    api_key = request.api_key.strip()

    if not api_key:
        return ValidateKeyResponse(valid=False, message="API key is required")

    validator = _get_validator(provider)
    if validator is None:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        return ValidateKeyResponse(
            valid=False,
            message=f"Unsupported provider '{request.provider}'. Supported providers: {supported}",
        )

    try:
        return await validator(api_key)
    except httpx.TimeoutException:
        return ValidateKeyResponse(valid=False, message=f"{provider} validation timed out")
    except httpx.HTTPError as exc:
        return ValidateKeyResponse(valid=False, message=f"{provider} validation error: {str(exc)[:200]}")
    except Exception as exc:  # nosec B110 - validation should degrade to a structured result
        return ValidateKeyResponse(valid=False, message=f"{provider} validation error: {str(exc)[:200]}")