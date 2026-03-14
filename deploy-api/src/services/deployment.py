"""Deployment orchestration: validate HF -> select GPU -> create Runpod endpoint -> notify user."""

import asyncio
import re
from datetime import UTC, datetime

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
from src.services.gpu_selection import select_gpu_candidates
from src.services.huggingface import validate_model
from src.services.internal_urls import build_deployment_ready_url, build_log_tunnel_url
from src.services.provider_factory import get_provider
from src.services.r2_manifest import (
    fetch_cached_model_ids,
    model_s3_url,
    split_s3_url,
)
from src.services.runpod import create_serverless_template
from src.services.secret_cache import get_secrets
from src.services.webhook import notify
from src.services.worker_routing import resolve_worker_target


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


def _format_log_message(message: str, **kwargs: object) -> str:
    fields = []
    for key, value in kwargs.items():
        if value in (None, "", [], {}, ()):  # keep Firestore log lines compact
            continue
        fields.append(f"{key}={value}")
    if not fields:
        return message
    return f"{message} [{', '.join(fields)}]"


def _execution_timeout_ms(settings, worker_profile: str) -> int:
    if worker_profile == "video":
        return settings.runpod_execution_timeout_ms_video
    return settings.runpod_execution_timeout_ms


def _runtime_hf_model_id(hf_model_id: str) -> str:
    runtime_aliases = {
        "Wan-AI/Wan2.1-T2V-1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    }
    return runtime_aliases.get(hf_model_id, hf_model_id)


def _uses_health_probe_readiness(worker_profile: str) -> bool:
    return True


def _uses_shared_model_cache(worker_profile: str) -> bool:
    return True


def _workers_min(settings, worker_profile: str, workers_max: int) -> int:
    if worker_profile == "video":
        return min(settings.runpod_workers_min_video, workers_max)
    return min(settings.runpod_workers_min, workers_max)


def _idle_timeout(settings, worker_profile: str) -> int:
    if worker_profile == "video":
        return settings.runpod_idle_timeout_seconds_video
    return settings.runpod_idle_timeout_seconds


def _model_load_wait_timeout_seconds(settings, worker_profile: str) -> int | None:
    if worker_profile != "video":
        return None
    execution_timeout_seconds = max(settings.runpod_execution_timeout_ms_video // 1000, 1)
    return max(execution_timeout_seconds - 60, 60)


def _runpod_init_timeout_seconds(settings, worker_profile: str) -> int | None:
    if worker_profile != "video":
        return None
    return max(settings.runpod_execution_timeout_ms_video // 1000, 1)


def _container_disk_gb(worker_profile: str) -> int:
    # Video image is ~13 GB; Wan models are ~10-11 GB + hf_transfer chunk headroom.
    # Image workers need ~25 GB (13 GB image + up to ~12 GB for SDXL / FLUX models).
    return 50 if worker_profile == "video" else 25


async def _create_deployment_template(
    *,
    api_key: str,
    worker_profile: str,
    image: str,
    deployment_id: str,
    env: list[dict[str, str]],
) -> tuple[str, str]:
    """Create a fresh per-deployment serverless template with env baked in.

    RunPod only injects TEMPLATE-level env as container OS env vars; endpoint-
    level env is metadata only. Using per-deployment templates avoids the shared
    template race condition while correctly propagating all deployment-specific
    env vars (HF_MODEL_ID, VISGATE_WEBHOOK, S3_MODEL_URL, etc.) to workers.

    Returns (template_id, template_name).
    """
    # Use a short but unique name derived from deployment_id
    template_name = f"visgate-dep-{deployment_id[-12:]}"
    result = await create_serverless_template(
        api_key=api_key,
        name=template_name,
        image_name=image,
        container_disk_in_gb=_container_disk_gb(worker_profile),
        env=env,
    )
    return result["id"], template_name

def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _as_run_url(endpoint_url: str | None) -> str | None:
    if not endpoint_url:
        return None
    return endpoint_url if endpoint_url.endswith("/run") else f"{endpoint_url}/run"


def _as_endpoint_root(endpoint_url: str | None) -> str | None:
    if not endpoint_url:
        return None
    return endpoint_url[:-4] if endpoint_url.endswith("/run") else endpoint_url


async def _probe_runpod_readiness(endpoint_url: str, api_key: str) -> tuple[bool, str | None]:
    """
    Probe worker readiness via the RunPod /health endpoint (GET, no job queued).
    Returns (ready, error_message).

    RunPod serverless workers don't start until a job arrives when workers_min=0.
    With workers_min>=1 the endpoint should eventually report ready/idle workers.
    We treat *any* successful /health response (HTTP < 400) with an initializing,
    ready, or idle worker as "ready", plus we accept the endpoint being live with
    zero workers when it has been responsive for the first time (endpoint exists
    and is routable).
    """
    endpoint_root = _as_endpoint_root(endpoint_url)
    if not endpoint_root:
        return False, "missing endpoint url"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{endpoint_root}/health",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code >= 400:
            return False, f"health probe http {resp.status_code}: {resp.text[:200]}"

        payload = resp.json()
        workers = payload.get("workers") or {}
        ready = int(workers.get("ready", 0))
        idle = int(workers.get("idle", 0))
        initializing = int(workers.get("initializing", 0))
        running = int(workers.get("running", 0))
        # Endpoint is ready if any worker exists (ready, idle, initializing, or running)
        if ready > 0 or idle > 0 or initializing > 0 or running > 0:
            return True, None
        # Endpoint responded successfully but no workers yet — still provisioning
        return False, None
    except Exception as exc:
        return False, str(exc)

async def orchestrate_deployment(
    deployment_id: str,
    runpod_api_key: str | None = None,
    hf_token_override: str | None = None,
) -> None:
    """
    Background task: validate HF model -> select GPU -> create cloud endpoint via provider.
    """
    settings = get_settings()
    client = get_firestore_client(settings.gcp_project_id)
    coll = settings.firestore_collection_deployments
    ctx = get_trace_context()

    def log_step(level: str, msg: str, **kwargs: object) -> None:
        formatted_message = _format_log_message(msg, **kwargs)
        structured_log(
            level,
            msg,
            deployment_id=deployment_id,
            operation="deployment.orchestrate",
            trace_id=ctx.get("trace_id"),
            span_id=ctx.get("span_id"),
            metadata=kwargs,
        )
        append_log(client, coll, deployment_id, level, formatted_message)

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

    def parse_worker_quota_limit(message: str) -> int | None:
        match = re.search(r"at most\s+(\d+)", message or "", flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return None

    try:
        doc = get_deployment(client, coll, deployment_id)
        if not doc:
            log_step("ERROR", "Deployment doc not found")
            return

        requested_hf_model_id = doc.hf_model_id
        runtime_hf_model_id = _runtime_hf_model_id(requested_hf_model_id)
        cached = None
        if not runpod_api_key:
            cached = get_secrets(deployment_id)
            runpod_api_key = cached.runpod_api_key if cached else None
        if not runpod_api_key:
            update_deployment(client, coll, deployment_id, {"status": "failed", "error": "Missing Runpod API key"})
            log_step("ERROR", "Missing Runpod API key for orchestration")
            return

        user_runpod_key = runpod_api_key
        gpu_tier = doc.gpu_tier
        hf_token: str | None = hf_token_override or (cached.hf_token if cached else None)
        if not hf_token:
            update_deployment(client, coll, deployment_id, {"status": "failed", "error": "Missing Hugging Face token"})
            log_step("ERROR", "Missing Hugging Face token for orchestration")
            return
        # Logic: Default to runpod for now, but could be fetched from doc metadata
        provider_name = "runpod"
        provider = get_provider(provider_name)

        worker_target = resolve_worker_target(settings, requested_hf_model_id, doc.task)

        # 1. Validate HF model + Check R2 cache in parallel
        update_deployment(client, coll, deployment_id, {"status": "validating"})
        log_step(
            "INFO",
            "Validating HF model and checking R2 cache in parallel",
            requested_hf_model_id=requested_hf_model_id,
            runtime_hf_model_id=runtime_hf_model_id,
        )

        async def _validate_hf():
            with span("deployment.validate_hf", {"hf_model_id": runtime_hf_model_id}):
                return await validate_model(runtime_hf_model_id, token=hf_token)

        async def _check_r2_cache():
            if not (settings.r2_access_key_id_rw and settings.r2_endpoint_url and _uses_shared_model_cache(worker_target["profile"])):
                return None, False
            model_bucket, _ = split_s3_url(settings.r2_model_base_url)
            per_model_url = model_s3_url(settings.r2_model_base_url, runtime_hf_model_id)
            cached_ids = fetch_cached_model_ids(
                settings.r2_endpoint_url,
                settings.r2_access_key_id_rw,
                settings.r2_secret_access_key_rw,
                bucket=model_bucket,
            )
            if runtime_hf_model_id in cached_ids:
                return per_model_url, False
            return None, True  # cache miss, trigger task

        model_info, (computed_s3_model_url, trigger_cache_model_task) = await asyncio.gather(
            _validate_hf(),
            _check_r2_cache(),
        )

        vram_gb = model_info.min_gpu_memory_gb
        update_deployment(client, coll, deployment_id, {"model_vram_gb": vram_gb})
        log_step(
            "INFO",
            "HF model validated",
            requested_hf_model_id=requested_hf_model_id,
            runtime_hf_model_id=runtime_hf_model_id,
            min_gpu_memory_gb=vram_gb,
        )
        if computed_s3_model_url:
            log_step("INFO", "R2 cache hit — worker will sync from R2", s3_url=computed_s3_model_url)
        elif trigger_cache_model_task:
            log_step("INFO", "R2 cache miss — separate job will download and cache")

        # 2. Select GPU — prefer live RunPod registry, fallback to Firestore, then static defaults
        update_deployment(client, coll, deployment_id, {"status": "selecting_gpu"})

        from src.services.gpu_registry import derive_tier_mapping, fetch_live_gpu_registry

        live_registry = await fetch_live_gpu_registry(user_runpod_key)
        if live_registry:
            registry = live_registry
            tier_mapping = derive_tier_mapping(live_registry)
            log_step("INFO", f"Using live RunPod GPU registry ({len(registry)} GPUs)")
        else:
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

        env = {
            "HF_MODEL_ID": runtime_hf_model_id,
            "VISGATE_DEPLOYMENT_ID": deployment_id,
            "RETURN_BASE64": "false",
        }
        if doc.task:
            env["TASK"] = doc.task
        if hf_token:
            env["HF_TOKEN"] = hf_token

        # Worker receives the platform R2 read-only key for model and staged-input reads.
        effective_access_key = settings.r2_access_key_id_ro or ""
        effective_secret_key = settings.r2_secret_access_key_ro or ""
        effective_endpoint = settings.r2_endpoint_url
        effective_s3_model_url = computed_s3_model_url or ""

        # Use VISGATE_R2_* prefixed names instead of standard AWS_* to prevent
        # RunPod from overriding them with its own internal AWS credentials.
        # loader.py reads these and injects them explicitly into the s5cmd subprocess env.
        if effective_access_key:
            env["VISGATE_R2_ACCESS_KEY_ID"] = effective_access_key
        if effective_secret_key:
            env["VISGATE_R2_SECRET_ACCESS_KEY"] = effective_secret_key
        if effective_endpoint:
            env["VISGATE_R2_ENDPOINT_URL"] = effective_endpoint
        if effective_s3_model_url:
            env["S3_MODEL_URL"] = effective_s3_model_url

        # Worker needs its own RunPod key to self-cleanup when idle
        env["RUNPOD_API_KEY"] = user_runpod_key

        internal_base = (doc.internal_webhook_base_url or getattr(settings, "internal_webhook_base_url", "") or "").rstrip("/")
        # Internal secret travels via header (X-Visgate-Internal-Secret), never in URL
        # to prevent it appearing in Cloud Run access logs.
        visgate_webhook = build_deployment_ready_url(internal_base, deployment_id)
        if visgate_webhook:
            env["VISGATE_WEBHOOK"] = visgate_webhook
        if settings.internal_webhook_secret:
            env["VISGATE_INTERNAL_SECRET"] = settings.internal_webhook_secret
        if settings.cleanup_idle_timeout_seconds:
            env["CLEANUP_IDLE_TIMEOUT_SECONDS"] = str(settings.cleanup_idle_timeout_seconds)
        if settings.cleanup_failure_threshold:
            env["CLEANUP_FAILURE_THRESHOLD"] = str(settings.cleanup_failure_threshold)
        log_tunnel_url = build_log_tunnel_url(internal_base, deployment_id)
        if log_tunnel_url:
            env["VISGATE_LOG_TUNNEL"] = log_tunnel_url

        # Common data for all providers
        endpoint_name = doc.endpoint_name or f"visgate-{deployment_id}"
        locations = (doc.region or settings.runpod_default_locations).strip()
        execution_timeout_ms = _execution_timeout_ms(settings, worker_target["profile"])
        model_load_wait_timeout_seconds = _model_load_wait_timeout_seconds(settings, worker_target["profile"])
        if model_load_wait_timeout_seconds is not None:
            env["MODEL_LOAD_WAIT_TIMEOUT_SECONDS"] = str(model_load_wait_timeout_seconds)
        runpod_init_timeout_seconds = _runpod_init_timeout_seconds(settings, worker_target["profile"])
        if runpod_init_timeout_seconds is not None:
            env["RUNPOD_INIT_TIMEOUT"] = str(runpod_init_timeout_seconds)

        # 3. Create Cloud Endpoint
        update_deployment(
            client,
            coll,
            deployment_id,
            {
                "status": "creating_endpoint",
                "worker_profile": worker_target["profile"],
                "worker_image": worker_target["image"],
            },
        )

        # MULTI-GPU TARGETING: Select top 5 candidates to maximise GPU availability
        target_candidates = gpu_candidates[:5]
        target_ids = [c[0] for c in target_candidates]
        target_display = ", ".join([c[1] for c in target_candidates])

        log_step(
            "INFO",
            f"Deploying to GPU pool: {target_display}",
            target_ids=target_ids,
            worker_profile=worker_target["profile"],
            image=worker_target["image"],
        )
        # Build the env list for the per-deployment template (RunPod only injects
        # template-level env as container OS env vars; endpoint env is metadata only)
        runpod_template_env = [{"key": k, "value": str(v)} for k, v in env.items()]
        dep_template_id, dep_template_name = await _create_deployment_template(
            api_key=user_runpod_key,
            worker_profile=worker_target["profile"],
            image=worker_target["image"],
            deployment_id=deployment_id,
            env=runpod_template_env,
        )
        update_deployment(client, coll, deployment_id, {"runpod_dep_template_name": dep_template_name})
        log_step(
            "INFO",
            "Created per-deployment RunPod template with env",
            worker_profile=worker_target["profile"],
            dep_template_id=dep_template_id,
            dep_template_name=dep_template_name,
            image=worker_target["image"],
        )

        async def create_endpoint_for_gpu_ids(gpu_ids: list[str], workers_max: int):
            try:
                return await provider.create_endpoint(
                    name=endpoint_name,
                    gpu_ids=gpu_ids,
                    image=worker_target["image"],
                    env={},
                    api_key=user_runpod_key,
                    template_id=dep_template_id,
                    execution_timeout_ms=execution_timeout_ms,
                    workers_min=_workers_min(settings, worker_target["profile"], workers_max),
                    workers_max=workers_max,
                    idle_timeout=_idle_timeout(settings, worker_target["profile"]),
                    scaler_type=settings.runpod_scaler_type,
                    scaler_value=settings.runpod_scaler_value,
                    volume_in_gb=settings.runpod_volume_size_gb,
                    locations=locations,
                )
            except Exception as exc:
                allowed_workers_max = parse_worker_quota_limit(str(exc))
                if allowed_workers_max is None or allowed_workers_max >= workers_max:
                    raise

                log_step(
                    "WARNING",
                    "RunPod worker quota reduced autoscaling; retrying with a lower worker cap",
                    requested_workers_max=workers_max,
                    adjusted_workers_max=allowed_workers_max,
                )
                return await provider.create_endpoint(
                    name=endpoint_name,
                    gpu_ids=gpu_ids,
                    image=worker_target["image"],
                    env={},
                    api_key=user_runpod_key,
                    template_id=dep_template_id,
                    execution_timeout_ms=execution_timeout_ms,
                    workers_min=_workers_min(settings, worker_target["profile"], allowed_workers_max),
                    workers_max=allowed_workers_max,
                    idle_timeout=_idle_timeout(settings, worker_target["profile"]),
                    scaler_type=settings.runpod_scaler_type,
                    scaler_value=settings.runpod_scaler_value,
                    volume_in_gb=settings.runpod_volume_size_gb,
                    locations=locations,
                )

        endpoint_data = None
        last_error = None
        try:
            endpoint_data = await create_endpoint_for_gpu_ids(
                target_ids,
                settings.runpod_workers_max,
            )
        except Exception as e:
            last_error = e
            if is_capacity_error(str(e)):
                log_step("WARNING", "Out of capacity for entire pool; falling back to global search")
                # Fallback to all candidates if top 3 fail (more expensive/rare ones)
                full_ids = [c[0] for c in gpu_candidates]
                endpoint_data = await create_endpoint_for_gpu_ids(
                    full_ids,
                    settings.runpod_workers_max,
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
                "worker_profile": worker_target["profile"],
                "worker_template_id": dep_template_id,
                "worker_image": worker_target["image"],
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
                    hf_model_id=runtime_hf_model_id,
                    hf_token=hf_token,
                )
                log_step("INFO", "Enqueued background cache modeling task")
            except Exception as exc:
                log_step("WARNING", f"Failed to enqueue cache modeling task: {exc}")

        # Fallback readiness monitor:
        # If worker webhook fails, probe RunPod /health directly (GET, no job queued).
        monitor_timeout_seconds = 600   # 10 min — covers GPU cold-start + model download
        monitor_interval_seconds = 5    # Poll every 5s (less aggressive, still responsive)
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
                    "ERROR",
                    "Readiness monitor timed out; marking deployment failed",
                    timeout_seconds=monitor_timeout_seconds,
                )
                update_deployment(
                    client,
                    coll,
                    deployment_id,
                    {"status": "failed", "error": f"Worker did not become ready within {monitor_timeout_seconds}s"},
                )
                return

            if _uses_health_probe_readiness(worker_target["profile"]):
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
    endpoint_url: str | None = None,
    t_r2_sync_s: float | None = None,
    t_model_load_s: float | None = None,
    loaded_from_cache: bool | None = None,
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
    if t_r2_sync_s is not None:
        updates["t_r2_sync_s"] = t_r2_sync_s
    if t_model_load_s is not None:
        updates["t_model_load_s"] = t_model_load_s
    if loaded_from_cache is not None:
        updates["loaded_from_cache"] = loaded_from_cache
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
        "t_r2_sync_s": t_r2_sync_s,
        "t_model_load_s": t_model_load_s,
        "loaded_from_cache": loaded_from_cache,
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
    success = True
    if doc.user_webhook_url:
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
    message: str | None = None,
    endpoint_url: str | None = None,
    t_r2_sync_s: float | None = None,
    t_model_load_s: float | None = None,
    loaded_from_cache: bool | None = None,
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
        return await mark_deployment_ready_and_notify(
            deployment_id,
            endpoint_url=endpoint_url,
            t_r2_sync_s=t_r2_sync_s,
            t_model_load_s=t_model_load_s,
            loaded_from_cache=loaded_from_cache,
        )

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
