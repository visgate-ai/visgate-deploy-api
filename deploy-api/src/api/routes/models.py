"""GET /v1/models — public model catalog endpoint."""

from __future__ import annotations

import logging
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import RequestContext, get_request_context
from src.core.config import get_settings
from src.models.model_specs_registry import MODEL_SPECS_REGISTRY
from src.models.schemas import HFModelResult, HFModelSearchResponse, ModelEntry, ModelsListResponse
from src.services.r2_manifest import fetch_cached_model_ids, split_s3_url

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
        settings.r2_endpoint_url
        and settings.r2_access_key_id_ro
        and settings.r2_secret_access_key_ro
    ):
        model_bucket, _ = split_s3_url(settings.r2_model_base_url)
        cached_ids = fetch_cached_model_ids(
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id_ro,
            secret_access_key=settings.r2_secret_access_key_ro,
            bucket=model_bucket,
        )
    else:
        logger.debug("R2 read-only credentials not configured; skipping manifest fetch")

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
        cache_enabled=bool(cached_ids or settings.r2_access_key_id_ro),
    )


@router.get("/search", response_model=HFModelSearchResponse, summary="Search HuggingFace models")
async def search_models(
    q: str,
    task: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    ctx: Annotated[RequestContext, Depends(get_request_context)] = None,
) -> HFModelSearchResponse:
    """Search for models on HuggingFace."""
    params: dict[str, Any] = {
        "search": q,
        "sort": "downloads",
        "direction": "-1",
        "limit": limit,
        "full": "false",
    }
    if task:
        params["pipeline_tag"] = task

    headers: dict[str, str] = {}
    if ctx and ctx.hf_token:
        headers["Authorization"] = f"Bearer {ctx.hf_token}"

    hf_url = "https://huggingface.co/api/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(hf_url, params=params, headers=headers)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "HF_UNREACHABLE", "message": f"HuggingFace API unreachable: {str(e)[:200]}"},
        )

    if resp.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail={"error": "HF_RATE_LIMIT", "message": "HuggingFace API rate limit hit."},
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={"error": "HF_ERROR", "message": f"HuggingFace returned {resp.status_code}"},
        )

    models = resp.json()
    results: list[HFModelResult] = []
    for m in models:
        results.append(
            HFModelResult(
                model_id=m.get("id", ""),
                task=m.get("pipeline_tag"),
                downloads=m.get("downloads") or 0,
                likes=m.get("likes") or 0,
            )
        )
    return HFModelSearchResponse(results=results, query=q)
