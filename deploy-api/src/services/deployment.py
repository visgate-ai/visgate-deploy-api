"""Deployment orchestration: validate HF -> select GPU -> create Runpod endpoint -> notify user."""

import asyncio
from datetime import UTC, datetime
from typing import Optional

import httpx

from src.core.config import get_settings
from src.core.errors import (
    HuggingFaceModelNotFoundError,
    RunpodAPIError,
    RunpodInsufficientGPUError,
)
from src.core.logging import structured_log
from src.core.telemetry import (
    get_trace_context,
    record_deployment_ready_duration,
    record_webhook_failure,
    span,
)
from src.services.firestore_repo import (
    append_log,
    get_deployment,
    get_firestore_client,
    get_gpu_registry,
    get_tier_mapping,
    update_deployment,
)
from src.services.gpu_selection import select_gpu
from src.services.gpu_selection import select_gpu_candidates
from src.services.huggingface import validate_model
from src.services.provider_factory import get_provider
import src.services.runpod # Trigger registration
from src.services.secret_cache import get_secrets
from src.services.webhook import notify

def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _as_run_url(endpoint_url: Optional[str]) -> Optional[str]:
    if not endpoint_url:
        return None
    return endpoint_url if endpoint_url.endswith("/run") else f"{endpoint_url}/run"


def _as_endpoint_root(endpoint_url: Optional[str]) -> Optional[str]:
    if not endpoint_url:
        return None
    return endpoint_url[:-4] if endpoint_url.endswith("/run") else endpoint_url


async def _probe_runpod_readiness(endpoint_url: str, api_key: str) -> tuple[bool, Optional[str]]:
    """
    Probe worker readiness with debug payload.
    Returns (ready, error_message).
    """
    endpoint_root = _as_endpoint_root(endpoint_url)
    if not endpoint_root:
        return False, "missing endpoint url"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{endpoint_root}/runsync",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": {"debug": True}},
            )
        if resp.status_code >= 400:
            return False, f"probe http {resp.status_code}: {resp.text[:200]}"

        payload = resp.json()
        status = str(payload.get("status", "")).upper()
        pipeline_loaded = bool(payload.get("pipeline_loaded"))
        error_text = str(payload.get("error", ""))

        if status == "OK" and pipeline_loaded:
            return True, None
        if status in {"IN_QUEUE", "IN_PROGRESS", "RUNNING", "LOADING"}:
            return False, None
        if "still loading" in error_text.lower():
            return False, None
        if status == "FAILED":
            return False, error_text or "worker reported failed status"
        return False, None
    except Exception as exc:
        return False, str(exc)

async def orchestrate_deployment(
    deployment_id: str,
    runpod_api_key: Optional[str] = None,
    hf_token_override: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_endpoint_url: Optional[str] = None,
    s3_model_url: Optional[str] = None,
) -> None:
    """
    Background task: validate HF model -> select GPU -> create cloud endpoint via provider.
    """
    settings = get_settings()
    client = get_firestore_client(settings.gcp_project_id)
    coll = settings.firestore_collection_deployments
    ctx = get_trace_context()

    def log_step(level: str, msg: str, **kwargs: object) -> None:
        structured_log(
            level,
            msg,
            deployment_id=deployment_id,
            operation="deployment.orchestrate",
            trace_id=ctx.get("trace_id"),
            span_id=ctx.get("span_id"),
            metadata=kwargs,
        )
        append_log(client, coll, deployment_id, level, msg)

    def is_capacity_error(message: str) -> bool:
        m = (message or "").lower()
        markers = [
            "insufficient",
            "no gpu",
            "no capacity",
            "out of capacity",
            "unavailable",
            "stock",
            "resource exhausted",
        ]
        return any(marker in m for marker in markers)

    try:
        doc = get_deployment(client, coll, deployment_id)
        if not doc:
            log_step("ERROR", "Deployment doc not found")
            return

        hf_model_id = doc.hf_model_id
        cached = None
        if not runpod_api_key:
            cached = get_secrets(deployment_id)
            runpod_api_key = cached.runpod_api_key if cached else None
        if cached:
            if aws_access_key_id is None:
                aws_access_key_id = cached.aws_access_key_id
            if aws_secret_access_key is None:
                aws_secret_access_key = cached.aws_secret_access_key
            if aws_endpoint_url is None:
                aws_endpoint_url = cached.aws_endpoint_url
            if s3_model_url is None:
                s3_model_url = cached.s3_model_url
        if not runpod_api_key:
            update_deployment(client, coll, deployment_id, {"status": "failed", "error": "Missing Runpod API key"})
            log_step("ERROR", "Missing Runpod API key for orchestration")
            return

        user_runpod_key = runpod_api_key
        gpu_tier = doc.gpu_tier
        hf_token: Optional[str] = hf_token_override or (cached.hf_token if cached else None)
        # Logic: Default to runpod for now, but could be fetched from doc metadata
        provider_name = "runpod" 
        provider = get_provider(provider_name)

        # 1. Validate HF model
        update_deployment(client, coll, deployment_id, {"status": "validating"})
        log_step("INFO", "Validating Hugging Face model")
        with span("deployment.validate_hf", {"hf_model_id": hf_model_id}):
            model_info = await validate_model(hf_model_id, token=hf_token)
        vram_gb = model_info.vram_gb
        update_deployment(client, coll, deployment_id, {"model_vram_gb": vram_gb})
        log_step("INFO", "HF model validated", hf_model_id=hf_model_id, vram_gb=vram_gb)

        # 2. Select GPU
        update_deployment(client, coll, deployment_id, {"status": "selecting_gpu"})
        registry = get_gpu_registry(client, settings.firestore_collection_gpu_registry)
        tier_mapping = get_tier_mapping(client, settings.firestore_collection_gpu_tiers)
        gpu_candidates = select_gpu_candidates(vram_gb, gpu_tier, registry=registry, tier_mapping=tier_mapping)
        gpu_id, gpu_display = gpu_candidates[0]
        log_step(
            "INFO",
            f"Selected GPU candidates count={len(gpu_candidates)}",
            first_gpu=gpu_display,
            provider=provider_name,
        )

        # 3. Create Cloud Endpoint
        update_deployment(client, coll, deployment_id, {"status": "creating_endpoint"})
        
        env = {
            "HF_MODEL_ID": hf_model_id,
            "VISGATE_DEPLOYMENT_ID": deployment_id,
        }
        if hf_token:
            env["HF_TOKEN"] = hf_token
        
        # Add AWS/S3 credentials for optimized loader
        effective_access_key = aws_access_key_id or settings.aws_access_key_id
        effective_secret_key = aws_secret_access_key or settings.aws_secret_access_key
        effective_endpoint = aws_endpoint_url or settings.aws_endpoint_url
        effective_s3_model_url = s3_model_url or settings.s3_model_url

        if effective_access_key:
            env["AWS_ACCESS_KEY_ID"] = effective_access_key
        if effective_secret_key:
            env["AWS_SECRET_ACCESS_KEY"] = effective_secret_key
        if effective_endpoint:
            env["AWS_ENDPOINT_URL"] = effective_endpoint
        if effective_s3_model_url:
            env["S3_MODEL_URL"] = effective_s3_model_url

        internal_base = getattr(settings, "internal_webhook_base_url", "") or ""
        visgate_webhook = f"{internal_base}/internal/deployment-ready/{deployment_id}"
        if settings.internal_webhook_secret:
            visgate_webhook += f"?secret={settings.internal_webhook_secret}"
        env["VISGATE_WEBHOOK"] = visgate_webhook
        if settings.internal_webhook_secret:
            env["VISGATE_INTERNAL_SECRET"] = settings.internal_webhook_secret
        if settings.cleanup_idle_timeout_seconds:
            env["CLEANUP_IDLE_TIMEOUT_SECONDS"] = str(settings.cleanup_idle_timeout_seconds)
        if settings.cleanup_failure_threshold:
            env["CLEANUP_FAILURE_THRESHOLD"] = str(settings.cleanup_failure_threshold)
        if internal_base:
            env["VISGATE_LOG_TUNNEL"] = f"{internal_base}/internal/logs/{deployment_id}"

        # Common data for all providers
        endpoint_name = doc.endpoint_name or f"visgate-{deployment_id}"
        locations = (doc.region or settings.runpod_default_locations).strip()
        endpoint_data = None
        last_error: Exception | None = None
        for candidate_id, candidate_display in gpu_candidates:
            try:
                endpoint_data = await provider.create_endpoint(
                    name=endpoint_name,
                    gpu_id=candidate_id,
                    image=settings.docker_image,
                    env=env,
                    api_key=user_runpod_key,
                    # Runpod specific kwargs
                    template_id=settings.runpod_template_id,
                    workers_min=1,
                    workers_max=2,
                    idle_timeout=300,
                    scaler_value=2,
                    volume_in_gb=settings.runpod_volume_size_gb,
                    locations=locations,
                )
                gpu_id = candidate_id
                gpu_display = candidate_display
                break
            except RunpodAPIError as exc:
                last_error = exc
                if is_capacity_error(exc.message):
                    log_step("WARNING", f"GPU candidate unavailable: {candidate_display}", gpu_id=candidate_id)
                    continue
                raise

        if endpoint_data is None:
            if last_error:
                raise last_error
            raise RunpodAPIError("No suitable GPU candidate endpoint could be created")
        
        endpoint_id = endpoint_data["id"]
        endpoint_url = endpoint_data["url"]
        
        update_deployment(
            client,
            coll,
            deployment_id,
            {
                "runpod_endpoint_id": endpoint_id,
                "endpoint_url": endpoint_url,
                "gpu_allocated": gpu_display,
                "provider": provider_name
            },
        )
        log_step(
            "INFO",
            f"{provider_name.capitalize()} endpoint created",
            endpoint_id=endpoint_id,
            endpoint_url=endpoint_url,
        )
        update_deployment(client, coll, deployment_id, {"status": "loading_model"})
        log_step("INFO", "Waiting for model load signal from worker")

        # Fallback readiness monitor:
        # If worker webhook fails, probe Runpod endpoint directly and mark ready.
        monitor_timeout_seconds = 900
        monitor_interval_seconds = 8
        started_at = asyncio.get_running_loop().time()
        while True:
            latest = get_deployment(client, coll, deployment_id)
            if not latest:
                log_step("WARNING", "Deployment doc missing during readiness monitoring")
                return
            if latest.status in {"ready", "failed", "webhook_failed", "deleted"}:
                return

            elapsed = asyncio.get_running_loop().time() - started_at
            if elapsed > monitor_timeout_seconds:
                log_step(
                    "WARNING",
                    "Readiness monitor timed out; waiting for worker webhook",
                    timeout_seconds=monitor_timeout_seconds,
                )
                return

            ready, probe_error = await _probe_runpod_readiness(endpoint_url, user_runpod_key)
            if ready:
                log_step("INFO", "Readiness probe succeeded; marking deployment ready")
                await mark_deployment_ready_and_notify(
                    deployment_id,
                    endpoint_url=_as_run_url(endpoint_url),
                )
                return
            if probe_error:
                log_step("WARNING", "Readiness probe retry", error=probe_error)
            await asyncio.sleep(monitor_interval_seconds)
    except HuggingFaceModelNotFoundError as e:
        update_deployment(
            client,
            coll,
            deployment_id,
            {"status": "failed", "error": e.message},
        )
        log_step("ERROR", e.message, error_type="HuggingFaceModelNotFoundError")
    except RunpodInsufficientGPUError as e:
        update_deployment(
            client,
            coll,
            deployment_id,
            {"status": "failed", "error": e.message},
        )
        log_step("ERROR", e.message, error_type="RunpodInsufficientGPUError")
    except RunpodAPIError as e:
        update_deployment(
            client,
            coll,
            deployment_id,
            {"status": "failed", "error": e.message},
        )
        log_step("ERROR", e.message, error_type="RunpodAPIError")
    except Exception as e:
        update_deployment(
            client,
            coll,
            deployment_id,
            {"status": "failed", "error": str(e)},
        )
        log_step("ERROR", str(e), error_type=type(e).__name__)


async def mark_deployment_ready_and_notify(
    deployment_id: str,
    endpoint_url: Optional[str] = None,
) -> bool:
    """
    Called from internal webhook handler: set status=ready, ready_at, then notify user.
    Returns True if user webhook was delivered.
    """
    settings = get_settings()
    client = get_firestore_client(settings.gcp_project_id)
    coll = settings.firestore_collection_deployments
    doc = get_deployment(client, coll, deployment_id)
    if not doc:
        return False

    if doc.status == "ready" and doc.ready_at:
        return True

    created_at = doc.created_at
    now = _now_iso()
    resolved_endpoint_url = _as_run_url(endpoint_url) or _as_run_url(doc.endpoint_url)
    updates: dict = {"status": "ready", "ready_at": now}
    if resolved_endpoint_url:
        updates["endpoint_url"] = resolved_endpoint_url
    update_deployment(client, coll, deployment_id, updates)
    append_log(
        client,
        coll,
        deployment_id,
        "INFO",
        "Model loaded, deployment ready",
    )

    # Duration metric
    duration_seconds = 0.0
    try:
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        ready_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
        duration_seconds = (ready_dt - created_dt).total_seconds()
        record_deployment_ready_duration(duration_seconds)
    except Exception:
        pass

    payload = {
        "event": "deployment_ready",
        "deployment_id": deployment_id,
        "status": "ready",
        "endpoint_url": resolved_endpoint_url,
        "runpod_endpoint_id": doc.runpod_endpoint_id,
        "model_id": doc.hf_model_id,
        "gpu_allocated": doc.gpu_allocated,
        "created_at": created_at,
        "ready_at": now,
        "duration_seconds": duration_seconds,
        "usage_example": {
            "method": "POST",
            "url": resolved_endpoint_url,
            "headers": {
                "Authorization": "Bearer <YOUR_RUNPOD_API_KEY>"
            },
            "body": {
                "input": {
                    "prompt": "An astronaut riding a horse in photorealistic style",
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5
                }
            }
        }
    }
    success = await notify(
        doc.user_webhook_url,
        payload,
        timeout_seconds=settings.webhook_timeout_seconds,
        retries=settings.webhook_max_retries,
        deployment_id=deployment_id,
    )
    if not success:
        update_deployment(
            client,
            coll,
            deployment_id,
            {"error": "User webhook delivery failed after retries"},
        )
        append_log(
            client,
            coll,
            deployment_id,
            "WARNING",
            "User webhook delivery failed after retries; deployment remains ready",
        )
        record_webhook_failure()
    return success


async def update_deployment_phase_from_worker(
    deployment_id: str,
    status: str,
    message: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> bool:
    """
    Update deployment for worker-reported intermediate/failure phases.
    """
    settings = get_settings()
    client = get_firestore_client(settings.gcp_project_id)
    coll = settings.firestore_collection_deployments
    doc = get_deployment(client, coll, deployment_id)
    if not doc:
        return False

    if status == "ready":
        return await mark_deployment_ready_and_notify(deployment_id, endpoint_url=endpoint_url)

    if doc.status == "ready" and status != "failed":
        return True

    updates: dict[str, object] = {"status": status}
    resolved_endpoint_url = _as_run_url(endpoint_url)
    if resolved_endpoint_url:
        updates["endpoint_url"] = resolved_endpoint_url
    if status == "failed":
        updates["error"] = message or "Worker reported failure"
    update_deployment(client, coll, deployment_id, updates)

    if message:
        log_message = message
    elif status == "downloading_model":
        log_message = "Worker is downloading model artifacts"
    elif status == "loading_model":
        log_message = "Worker is loading model into memory"
    elif status == "failed":
        log_message = "Worker reported model loading failure"
    else:
        log_message = f"Worker phase update: {status}"

    append_log(client, coll, deployment_id, "ERROR" if status == "failed" else "INFO", log_message)
    return True
