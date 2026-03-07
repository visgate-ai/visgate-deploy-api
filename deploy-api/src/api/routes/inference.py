"""Inference job lifecycle routes built on top of ready deployments."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, status

from src.api.dependencies import RequestContext, get_firestore, get_request_context
from src.core.config import get_settings
from src.core.errors import (
    DeploymentNotFoundError,
    InferenceJobNotFoundError,
    InvalidInferenceJobRequestError,
)
from src.models.entities import InferenceJobDoc
from src.models.schemas import (
    InferenceJobAcceptedResponse,
    InferenceJobCreate,
    InferenceJobListResponse,
    InferenceJobResponse,
)
from src.services.inference_jobs import (
    build_job_metrics,
    compact_payload,
    estimate_cost_from_execution,
    extract_artifact_metadata,
    extract_estimated_cost,
    generate_job_id,
    map_provider_status,
    now_iso,
    parse_iso,
    resolve_gpu_hourly_price,
    sanitize_s3_config,
)
from src.services.internal_urls import build_inference_job_callback_url, resolve_internal_base_url
from src.services.model_capabilities import supports_task
from src.services.provider_factory import get_provider

router = APIRouter(prefix="/v1/inference/jobs", tags=["inference"])


def _get_repo():
    if get_settings().effective_use_memory_repo:
        import src.services.memory_repo as repo
    else:
        import src.services.firestore_repo as repo
    return repo


def _get_deployment(*args, **kwargs):
    return _get_repo().get_deployment(*args, **kwargs)


def _get_job(*args, **kwargs):
    return _get_repo().get_inference_job(*args, **kwargs)


def _set_job(*args, **kwargs):
    return _get_repo().set_inference_job(*args, **kwargs)


def _update_job(*args, **kwargs):
    return _get_repo().update_inference_job(*args, **kwargs)


def _list_jobs(*args, **kwargs):
    return _get_repo().list_inference_jobs(*args, **kwargs)


def _job_to_response(doc: InferenceJobDoc) -> InferenceJobResponse:
    created_at = parse_iso(doc.created_at)
    updated_at = parse_iso(doc.updated_at)
    completed_at = parse_iso(doc.completed_at)
    if created_at is None or updated_at is None:
        raise InvalidInferenceJobRequestError("Inference job timestamps are invalid", details={"job_id": doc.job_id})
    return InferenceJobResponse(
        job_id=doc.job_id,
        deployment_id=doc.deployment_id,
        provider=doc.provider,
        provider_job_id=doc.provider_job_id,
        task=doc.task,
        status=doc.status,
        provider_status=doc.provider_status,
        endpoint_url=doc.endpoint_url,
        input=doc.input_payload,
        output_destination=doc.output_destination,
        artifact=doc.artifact,
        metrics=doc.metrics,
        estimated_cost_usd=doc.estimated_cost_usd,
        output=doc.output_payload,
        output_preview=doc.output_preview,
        error=doc.error,
        progress=doc.progress,
        created_at=created_at,
        updated_at=updated_at,
        completed_at=completed_at,
        user_webhook_url=doc.user_webhook_url,
    )


def _build_internal_job_webhook(job_id: str, request: Request | None = None) -> str | None:
    settings = get_settings()
    if not settings.internal_webhook_secret:
        return None
    base = resolve_internal_base_url(request)
    url = build_inference_job_callback_url(base, job_id, settings.internal_webhook_secret)
    return url or None


async def _refresh_job_status(job_doc: InferenceJobDoc, api_key: str, firestore_client: Any) -> InferenceJobDoc:
    settings = get_settings()
    provider = get_provider(job_doc.provider)
    data = await provider.get_job_status(job_doc.endpoint_url, job_doc.provider_job_id, api_key)
    provider_status = data.get("status", job_doc.provider_status)
    status_value = map_provider_status(provider_status)
    updates: dict[str, Any] = {
        "provider_status": provider_status,
        "status": status_value,
        "updated_at": now_iso(),
        "progress": compact_payload(data.get("raw_response", {}).get("progress")),
    }
    if data.get("error") is not None:
        updates["error"] = compact_payload(data.get("error"))
    if data.get("output") is not None:
        updates["output_preview"] = compact_payload(data.get("output"))
        updates["artifact"] = extract_artifact_metadata(data.get("output"), job_doc.output_destination)
        updates["output_payload"] = compact_payload(data.get("output"))
    if status_value in {"completed", "failed", "cancelled", "expired"}:
        completed_at = now_iso()
        updates["completed_at"] = completed_at
        updates["metrics"] = build_job_metrics(
            created_at=job_doc.created_at,
            completed_at=completed_at,
            queue_ms=data.get("delay_time"),
            execution_ms=data.get("execution_time"),
        )
    elif data.get("delay_time") is not None or data.get("execution_time") is not None:
        updates["metrics"] = build_job_metrics(
            created_at=job_doc.created_at,
            completed_at=job_doc.completed_at,
            queue_ms=data.get("delay_time"),
            execution_ms=data.get("execution_time"),
        )
    estimated_cost_usd = extract_estimated_cost(data.get("raw_response"))
    if estimated_cost_usd is None and data.get("execution_time") is not None:
        estimated_cost_usd = estimate_cost_from_execution(data.get("execution_time"), job_doc.gpu_price_per_hour_usd)
    if estimated_cost_usd is not None:
        updates["estimated_cost_usd"] = estimated_cost_usd
    _update_job(firestore_client, settings.firestore_collection_inference_jobs, job_doc.job_id, updates)
    refreshed = _get_job(firestore_client, settings.firestore_collection_inference_jobs, job_doc.job_id)
    if not refreshed:
        raise InferenceJobNotFoundError(job_doc.job_id)
    return refreshed


@router.post("", response_model=InferenceJobAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_inference_job(
    request: Request,
    body: InferenceJobCreate,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> InferenceJobAcceptedResponse:
    settings = get_settings()
    deployment = _get_deployment(firestore_client, settings.firestore_collection_deployments, body.deployment_id)
    if not deployment or deployment.user_hash != ctx.user_hash:
        raise DeploymentNotFoundError(body.deployment_id)
    if deployment.status != "ready" or not deployment.endpoint_url:
        raise InvalidInferenceJobRequestError(
            "Deployment is not ready for inference jobs",
            details={"deployment_id": body.deployment_id, "status": deployment.status},
        )

    task = body.task or deployment.task or "text_to_image"
    if task and not supports_task(deployment.hf_model_id, task):
        raise InvalidInferenceJobRequestError(
            "Deployment model does not support requested task",
            details={"deployment_id": body.deployment_id, "task": task, "hf_model_id": deployment.hf_model_id},
        )

    provider = get_provider(deployment.provider or "runpod")
    job_id = generate_job_id()
    internal_webhook_url = _build_internal_job_webhook(job_id, request)
    provider_webhook = internal_webhook_url or (str(body.user_webhook_url) if body.user_webhook_url else None)
    created_at = now_iso()
    gpu_price_per_hour_usd = None
    if deployment.gpu_allocated:
        try:
            gpu_types = await provider.list_gpu_types(ctx.runpod_api_key)
            gpu_price_per_hour_usd = resolve_gpu_hourly_price(deployment.gpu_allocated, gpu_types)
        except Exception:
            gpu_price_per_hour_usd = None
    accepted = await provider.submit_job(
        deployment.endpoint_url,
        ctx.runpod_api_key,
        body.input,
        webhook_url=provider_webhook,
        policy=body.policy.model_dump(exclude_none=True) if body.policy else None,
        s3_config=body.s3_config.model_dump(exclude_none=True),
    )
    output_destination = sanitize_s3_config(body.s3_config.model_dump(exclude_none=True))

    doc = InferenceJobDoc(
        job_id=job_id,
        deployment_id=deployment.deployment_id,
        provider=deployment.provider or "runpod",
        provider_job_id=accepted["id"],
        endpoint_url=deployment.endpoint_url,
        status=map_provider_status(accepted.get("status")),
        provider_status=accepted.get("status", "IN_QUEUE"),
        gpu_allocated=deployment.gpu_allocated,
        gpu_price_per_hour_usd=gpu_price_per_hour_usd,
        hf_model_id=deployment.hf_model_id,
        task=task,
        input_payload=body.input,
        output_destination=output_destination,
        created_at=created_at,
        updated_at=created_at,
        user_hash=ctx.user_hash,
        user_webhook_url=str(body.user_webhook_url) if body.user_webhook_url else None,
    )
    _set_job(firestore_client, settings.firestore_collection_inference_jobs, doc)
    created_dt = parse_iso(created_at)
    if created_dt is None:
        raise InvalidInferenceJobRequestError("Inference job timestamp could not be parsed", details={"job_id": job_id})
    return InferenceJobAcceptedResponse(
        job_id=job_id,
        deployment_id=deployment.deployment_id,
        provider=doc.provider,
        provider_job_id=doc.provider_job_id,
        status=doc.status,
        provider_status=doc.provider_status,
        output_destination=output_destination,
        created_at=created_dt,
    )


@router.get("", response_model=InferenceJobListResponse)
async def list_inference_jobs(
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
    deployment_id: str | None = None,
    limit: int = 20,
) -> InferenceJobListResponse:
    settings = get_settings()
    docs = _list_jobs(
        firestore_client,
        settings.firestore_collection_inference_jobs,
        ctx.user_hash,
        deployment_id=deployment_id,
        limit=limit,
    )
    return InferenceJobListResponse(jobs=[_job_to_response(doc) for doc in docs], total=len(docs))


@router.get("/{job_id}", response_model=InferenceJobResponse)
async def get_inference_job(
    job_id: str,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
    refresh: bool = True,
) -> InferenceJobResponse:
    settings = get_settings()
    doc = _get_job(firestore_client, settings.firestore_collection_inference_jobs, job_id)
    if not doc or doc.user_hash != ctx.user_hash:
        raise InferenceJobNotFoundError(job_id)
    if refresh and doc.status not in {"completed", "failed", "cancelled", "expired"}:
        doc = await _refresh_job_status(doc, ctx.runpod_api_key, firestore_client)
    return _job_to_response(doc)


@router.post("/{job_id}/cancel", response_model=InferenceJobResponse)
async def cancel_inference_job(
    job_id: str,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> InferenceJobResponse:
    settings = get_settings()
    doc = _get_job(firestore_client, settings.firestore_collection_inference_jobs, job_id)
    if not doc or doc.user_hash != ctx.user_hash:
        raise InferenceJobNotFoundError(job_id)
    provider = get_provider(doc.provider)
    result = await provider.cancel_job(doc.endpoint_url, doc.provider_job_id, ctx.runpod_api_key)
    updates = {
        "provider_status": result.get("status", "CANCELLED"),
        "status": map_provider_status(result.get("status", "CANCELLED")),
        "updated_at": now_iso(),
        "completed_at": now_iso(),
    }
    _update_job(firestore_client, settings.firestore_collection_inference_jobs, job_id, updates)
    refreshed = _get_job(firestore_client, settings.firestore_collection_inference_jobs, job_id)
    if not refreshed:
        raise InferenceJobNotFoundError(job_id)
    return _job_to_response(refreshed)


@router.post("/{job_id}/retry", response_model=InferenceJobResponse)
async def retry_inference_job(
    job_id: str,
    ctx: Annotated[RequestContext, Depends(get_request_context)],
    firestore_client=Depends(get_firestore),
) -> InferenceJobResponse:
    settings = get_settings()
    doc = _get_job(firestore_client, settings.firestore_collection_inference_jobs, job_id)
    if not doc or doc.user_hash != ctx.user_hash:
        raise InferenceJobNotFoundError(job_id)
    provider = get_provider(doc.provider)
    result = await provider.retry_job(doc.endpoint_url, doc.provider_job_id, ctx.runpod_api_key)
    updates = {
        "provider_status": result.get("status", "IN_QUEUE"),
        "status": map_provider_status(result.get("status", "IN_QUEUE")),
        "updated_at": now_iso(),
        "completed_at": None,
        "error": None,
    }
    _update_job(firestore_client, settings.firestore_collection_inference_jobs, job_id, updates)
    refreshed = _get_job(firestore_client, settings.firestore_collection_inference_jobs, job_id)
    if not refreshed:
        raise InferenceJobNotFoundError(job_id)
    return _job_to_response(refreshed)