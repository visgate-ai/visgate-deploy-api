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
from src.services.gpu_selection import select_gpu
from src.services.gpu_selection import select_gpu_candidates
from src.services.huggingface import validate_model
from src.services.provider_factory import get_provider
import src.services.runpod  # Trigger provider registration
from src.services.r2_manifest import (
    fetch_cached_model_ids,
    model_s3_url,
)
from src.services.secret_cache import get_secrets
from src.services.webhook import notify


def _get_repo():
    """Return the active repo module (firestore or in-memory) based on settings."""
    if get_settings().effective_use_memory_repo:
        import src.services.memory_repo as r
    else:
        import src.services.firestore_repo as r
    return r


def get_firestore_client(*args, **kwargs):
    return _get_repo().get_firestore_client(*args, **kwargs)


def append_log(*args, **kwargs):
    return _get_repo().append_log(*args, **kwargs)


def get_deployment(*args, **kwargs):
    return _get_repo().get_deployment(*args, **kwargs)


def get_gpu_registry(*args, **kwargs):
    return _get_repo().get_gpu_registry(*args, **kwargs)


def get_tier_mapping(*args, **kwargs):
    return _get_repo().get_tier_mapping(*args, **kwargs)


def update_deployment(*args, **kwargs):
    return _get_repo().update_deployment(*args, **kwargs)

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
        job_status = str(payload.get("status", "")).upper()
        if job_status in {"IN_QUEUE", "IN_PROGRESS"}:
            return False, None
        if job_status == "FAILED":
            return False, str(payload.get("error", "RunPod job failed"))

        output = payload.get("output") or {}
        # Support fallback if structure is flat for some reason
        if not output and "pipeline_loaded" in payload:
            output = payload

        status = str(output.get("status", "")).upper()
        pipeline_loaded = bool(output.get("pipeline_loaded"))
        error_text = str(output.get("error", ""))

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
        hf_token: Optional[str] = hf_token_override or (cached.hf_token if cached else None) or settings.hf_pro_access_token
        # Logic: Default to runpod for now, but could be fetched from doc metadata
        provider_name = "runpod" 
        provider = get_provider(provider_name)

        # 1. Validate HF model
        update_deployment(client, coll, deployment_id, {"status": "validating"})
        log_step("INFO", "Validating Hugging Face model")
        with span("deployment.validate_hf", {"hf_model_id": hf_model_id}):
            model_info = await validate_model(hf_model_id, token=hf_token)
        vram_gb = model_info.min_gpu_memory_gb
        update_deployment(client, coll, deployment_id, {"model_vram_gb": vram_gb})
        log_step("INFO", "HF model validated", hf_model_id=hf_model_id, min_gpu_memory_gb=vram_gb)

        # 1.5. Check R2 model cache — if present, worker will download from R2
        # The platform RW key lives only in the API (never sent to workers).
        # computed_s3_model_url is set to the per-model R2 path and passed to
        # the worker's S3_MODEL_URL so it can sync from R2 instead of HF.
        # On cache miss: separate Cloud Task will download from HF and upload to R2.
        computed_s3_model_url: Optional[str] = None
        trigger_cache_model_task: bool = False
        if settings.aws_access_key_id and settings.aws_endpoint_url and not s3_model_url:
            per_model_url = model_s3_url(settings.s3_model_url, hf_model_id)
            update_deployment(client, coll, deployment_id, {"status": "checking_r2_cache"})
            log_step("INFO", "Checking R2 model cache", hf_model_id=hf_model_id)
            cached_ids = fetch_cached_model_ids(
                settings.aws_endpoint_url,
                settings.aws_access_key_id,
                settings.aws_secret_access_key,
            )
            if hf_model_id in cached_ids:
                computed_s3_model_url = per_model_url
                log_step("INFO", "R2 cache hit — worker will sync from R2", s3_url=computed_s3_model_url)
            else:
                log_step("INFO", "R2 cache miss — separate job will download and cache", hf_model_id=hf_model_id)
                trigger_cache_model_task = True

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
        if doc.task:
            env["TASK"] = doc.task
        if hf_token:
            env["HF_TOKEN"] = hf_token
        
        # AWS/S3 credentials injected into RunPod worker:
        #   - cache_scope=private  → user's own keys (passed via aws_access_key_id param)
        #   - cache_scope=shared   → platform R2 READ-ONLY key (RW key stays in API only)
        #   - cache_scope=off      → no keys
        # The RW platform key (settings.aws_access_key_id) is intentionally NEVER sent to workers.
        if aws_access_key_id:
            # User's private credentials
            effective_access_key = aws_access_key_id
            effective_secret_key = aws_secret_access_key or ""
        elif settings.r2_access_key_id_r:
            # Platform shared cache — read-only key only
            effective_access_key = settings.r2_access_key_id_r
            effective_secret_key = settings.r2_secret_access_key_r
        else:
            effective_access_key = ""
            effective_secret_key = ""

        effective_endpoint = aws_endpoint_url or settings.aws_endpoint_url
        # Use user-provided URL, or the per-model R2 URL computed from the cache check,
        # or fall back to empty string (worker downloads from HuggingFace directly).
        effective_s3_model_url = s3_model_url or computed_s3_model_url or ""

        if effective_access_key:
            env["AWS_ACCESS_KEY_ID"] = effective_access_key
        if effective_secret_key:
            env["AWS_SECRET_ACCESS_KEY"] = effective_secret_key
        if effective_endpoint:
            env["AWS_ENDPOINT_URL"] = effective_endpoint
        if effective_s3_model_url:
            env["S3_MODEL_URL"] = effective_s3_model_url

        # Worker needs its own RunPod key to self-cleanup when idle
        env["RUNPOD_API_KEY"] = user_runpod_key

        internal_base = getattr(settings, "internal_webhook_base_url", "") or ""
        # Internal secret travels via header (X-Visgate-Internal-Secret), never in URL
        # to prevent it appearing in Cloud Run access logs.
        visgate_webhook = f"{internal_base}/internal/deployment-ready/{deployment_id}"
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
        
        # MULTI-GPU TARGETING: Select top 3 candidates to increase availability
        target_candidates = gpu_candidates[:3]
        target_ids = [c[0] for c in target_candidates]
        target_display = ", ".join([c[1] for c in target_candidates])
        
        log_step(
            "INFO", 
            f"Deploying to GPU pool: {target_display}",
            target_ids=target_ids
        )

        endpoint_data = None
        last_error = None
        try:
            endpoint_data = await provider.create_endpoint(
                name=endpoint_name,
                gpu_ids=target_ids, # Pass a list of GPU IDs
                image=settings.docker_image,
                env=env,
                api_key=user_runpod_key,
                template_id=settings.runpod_template_id,
                workers_min=settings.runpod_workers_min,
                workers_max=settings.runpod_workers_max,
                idle_timeout=settings.runpod_idle_timeout_seconds,
                scaler_type=settings.runpod_scaler_type,
                scaler_value=settings.runpod_scaler_value,
                volume_in_gb=settings.runpod_volume_size_gb,
                locations=locations,
            )
        except Exception as e:
            last_error = e
            if is_capacity_error(str(e)):
                log_step("WARNING", "Out of capacity for entire pool; falling back to global search")
                # Fallback to all candidates if top 3 fail (more expensive/rare ones)
                full_ids = [c[0] for c in gpu_candidates]
                endpoint_data = await provider.create_endpoint(
                    name=endpoint_name,
                    gpu_ids=full_ids, # Pass a list of all GPU IDs
                    image=settings.docker_image,
                    env=env,
                    api_key=user_runpod_key,
                    template_id=settings.runpod_template_id,
                    workers_min=settings.runpod_workers_min,
                    workers_max=settings.runpod_workers_max,
                    idle_timeout=settings.runpod_idle_timeout_seconds,
                    scaler_type=settings.runpod_scaler_type,
                    scaler_value=settings.runpod_scaler_value,
                    volume_in_gb=settings.runpod_volume_size_gb,
                    locations=locations,
                )
            else:
                raise e

        if endpoint_data is None:
            if last_error:
                raise last_error
            raise RunpodAPIError("No suitable GPU candidate endpoint could be created")
        
        endpoint_id = endpoint_data["id"]
        endpoint_url = endpoint_data["url"]
        
        # 4. Finalize
        update_deployment(
            client,
            coll,
            deployment_id,
            {
                "status": "loading_model",
                "runpod_endpoint_id": endpoint_id,
                "endpoint_url": endpoint_url,
                "gpu_allocated": target_display,
                "provider": provider_name,
                "gpu_pool_targeted": target_ids,
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

        # Trigger background cache modeling job if needed (runs independently)
        if trigger_cache_model_task:
            try:
                from src.services.tasks import enqueue_cache_model_task
                await enqueue_cache_model_task(
                    hf_model_id=hf_model_id,
                    hf_token=hf_token,
                )
                log_step("INFO", "Enqueued background cache modeling task")
            except Exception as exc:
                log_step("WARNING", f"Failed to enqueue cache modeling task: {exc}")

        # Fallback readiness monitor:
        # If worker webhook fails, probe Runpod endpoint directly and mark ready.
        monitor_timeout_seconds = 900
        monitor_interval_seconds = 5  # 5s: faster webhook-fallback detection
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


def _inference_example_input(model_id: str) -> dict:
    """Return model-appropriate inference defaults for usage_example in webhook payload."""
    mid = (model_id or "").lower()
    # Turbo / distilled models: low steps, zero guidance
    if any(k in mid for k in ("turbo", "schnell", "lightning", "hyper")):
        return {"prompt": "An astronaut riding a horse in photorealistic style",
                "num_inference_steps": 4, "guidance_scale": 0.0, "width": 512, "height": 512}
    # FLUX-dev and similar: moderate steps, moderate guidance
    if "flux" in mid:
        return {"prompt": "An astronaut riding a horse in photorealistic style",
                "num_inference_steps": 28, "guidance_scale": 3.5, "width": 1024, "height": 1024}
    # Default: SDXL-base, SD2.x, SD3.x
    return {"prompt": "An astronaut riding a horse in photorealistic style",
            "num_inference_steps": 30, "guidance_scale": 7.5, "width": 1024, "height": 1024}


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
    except Exception:  # nosec B110 — metric failure must not block webhook dispatch
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
                "input": _inference_example_input(doc.hf_model_id or "")
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
