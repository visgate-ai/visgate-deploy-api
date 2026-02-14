"""Deployment orchestration: validate HF -> select GPU -> create Runpod endpoint -> notify user."""

import asyncio
from datetime import UTC, datetime
from typing import Optional

from google.cloud import firestore

from src.core.config import get_settings
from src.core.errors import (
    HuggingFaceModelNotFoundError,
    RunpodAPIError,
    RunpodInsufficientGPUError,
    WebhookDeliveryError,
)
from src.core.logging import structured_log
from src.core.telemetry import (
    get_trace_context,
    record_deployment_created,
    record_deployment_ready_duration,
    record_webhook_failure,
    span,
)
from src.models.entities import DeploymentDoc, LogEntry
from src.services.firestore_repo import (
    append_log,
    get_deployment,
    get_firestore_client,
    update_deployment,
)
from src.services.huggingface import validate_model
from src.services.runpod import create_endpoint, endpoint_run_url, select_gpu
from src.services.webhook import notify


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


async def orchestrate_deployment(deployment_id: str) -> None:
    """
    Background task: validate HF model -> select GPU -> create Runpod endpoint.
    Status 'ready' and user webhook are triggered by POST /internal/deployment-ready/{id}.
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

    try:
        doc = get_deployment(client, coll, deployment_id)
        if not doc:
            log_step("ERROR", "Deployment doc not found")
            return

        hf_model_id = doc.hf_model_id
        user_webhook_url = doc.user_webhook_url
        user_runpod_key = doc.user_runpod_key_ref  # stored as key or secret ref
        gpu_tier = doc.gpu_tier
        hf_token: Optional[str] = doc.hf_token_ref  # or resolve from Secret Manager

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
        gpu_id, gpu_display = select_gpu(vram_gb, gpu_tier)
        log_step("INFO", f"Selected GPU: {gpu_display}", gpu_id=gpu_id)

        # 3. Create Runpod endpoint
        update_deployment(client, coll, deployment_id, {"status": "creating_endpoint"})
        template_id = settings.runpod_template_id or "placeholder"
        if not template_id:
            update_deployment(
                client,
                coll,
                deployment_id,
                {"status": "failed", "error": "RUNPOD_TEMPLATE_ID not configured"},
            )
            log_step("ERROR", "RUNPOD_TEMPLATE_ID not set")
            return

        env = {
            "HF_MODEL_ID": hf_model_id,
        }
        if hf_token:
            env["HF_TOKEN"] = hf_token
        # Internal webhook URL: caller must set INTERNAL_WEBHOOK_BASE or we use relative
        internal_base = getattr(settings, "internal_webhook_base_url", "") or ""
        visgate_webhook = f"{internal_base}/internal/deployment-ready/{deployment_id}"
        env["VISGATE_WEBHOOK"] = visgate_webhook

        result = await create_endpoint(
            api_key=user_runpod_key,
            name=f"visgate-{deployment_id}",
            template_id=template_id,
            gpu_ids=gpu_id,
            env=env,
            url=settings.runpod_graphql_url,
            max_retries=settings.runpod_max_retries,
            workers_max=1,
        )
        runpod_endpoint_id = result.get("id")
        endpoint_url = endpoint_run_url(runpod_endpoint_id) if runpod_endpoint_id else None
        update_deployment(
            client,
            coll,
            deployment_id,
            {
                "runpod_endpoint_id": runpod_endpoint_id,
                "endpoint_url": endpoint_url,
                "gpu_allocated": gpu_display,
            },
        )
        log_step(
            "INFO",
            "Runpod endpoint created",
            runpod_endpoint_id=runpod_endpoint_id,
            endpoint_url=endpoint_url,
        )
        # Status stays creating_endpoint until container calls /internal/deployment-ready
        # (or we could set downloading_model / loading_model if container reports phases)
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

    created_at = doc.created_at
    now = _now_iso()
    updates: dict = {"status": "ready", "ready_at": now}
    if endpoint_url:
        updates["endpoint_url"] = endpoint_url
    update_deployment(client, coll, deployment_id, updates)
    append_log(
        client,
        coll,
        deployment_id,
        "INFO",
        "Model loaded, deployment ready",
    )

    # Duration metric
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
        "endpoint_url": doc.endpoint_url or endpoint_url,
        "runpod_endpoint_id": doc.runpod_endpoint_id,
        "model_id": doc.hf_model_id,
        "gpu_allocated": doc.gpu_allocated,
        "created_at": created_at,
        "ready_at": now,
        "duration_seconds": duration_seconds,
        "usage_example": {
            "method": "POST",
            "url": f"{doc.endpoint_url or endpoint_url}/run",
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
        update_deployment(client, coll, deployment_id, {"status": "webhook_failed"})
        record_webhook_failure()
    return success
