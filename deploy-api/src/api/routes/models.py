"""GET /v1/models — public model catalog endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.core.config import get_settings
from src.models.model_specs_registry import MODEL_SPECS_REGISTRY
from src.models.schemas import ModelEntry, ModelsListResponse
from src.services.r2_manifest import fetch_cached_model_ids

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=ModelsListResponse, summary="List available models")
async def list_models() -> ModelsListResponse:
    """Return the platform model catalogue.

    Each entry states the model ID, supported inference tasks, minimum GPU
    memory and whether the model weights are already cached in the platform R2
    storage layer (``cached=true`` means first-deploy cold-start is avoided).
    """
    settings = get_settings()

    # Attempt to read R2 manifest — graceful degradation if R2 not configured
    cached_ids: set[str] = set()
    if (
        settings.aws_endpoint_url
        and settings.r2_access_key_id_r
        and settings.r2_secret_access_key_r
    ):
        cached_ids = fetch_cached_model_ids(
            endpoint_url=settings.aws_endpoint_url,
            access_key_id=settings.r2_access_key_id_r,
            secret_access_key=settings.r2_secret_access_key_r,
        )
    else:
        logger.debug("R2 RO credentials not configured; skipping manifest fetch")

    entries = [
        ModelEntry(
            model_id=model_id,
            tasks=spec.get("tasks", []),
            gpu_memory_gb=spec.get("gpu_memory_gb", 0),
            cached=model_id in cached_ids,
        )
        for model_id, spec in MODEL_SPECS_REGISTRY.items()
    ]

    # Stable sort: cached first, then alphabetical
    entries.sort(key=lambda e: (not e.cached, e.model_id))

    return ModelsListResponse(
        models=entries,
        total=len(entries),
        cache_enabled=bool(cached_ids or settings.r2_access_key_id_r),
    )
