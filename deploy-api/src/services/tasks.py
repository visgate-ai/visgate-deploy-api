"""Google Cloud Tasks integration for background processing."""

import asyncio
import json

from google.cloud import tasks_v2
from src.core.config import get_settings
from src.core.logging import structured_log
from src.services.deployment import orchestrate_deployment


async def enqueue_orchestration_task(
    deployment_id: str,
    runpod_api_key: str,
    hf_token: str | None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_endpoint_url: str | None = None,
    s3_model_url: str | None = None,
) -> None:
    """
    Enqueue a task to orchestrate deployment.
    If cloud_tasks_queue_path is configured, uses Cloud Tasks.
    Otherwise, falls back to asyncio.create_task (not reliable for long ops in Cloud Run).
    """
    settings = get_settings()
    queue_path = settings.cloud_tasks_queue_path

    if settings.stateless_mode or not queue_path:
        structured_log(
            "WARNING",
            # F6: With stateless_mode=True and no Cloud Tasks queue, orchestration runs in-process
            # via asyncio.create_task. If the Cloud Run instance is scaled to zero while the task
            # is running, the deployment will stall silently. For production reliability, set
            # CLOUD_TASKS_QUEUE_PATH to use Cloud Tasks as a durable execution backend.
            "Stateless mode or missing Cloud Tasks; using in-process orchestration",
            deployment_id=deployment_id,
        )
        asyncio.create_task(
            orchestrate_deployment(
                deployment_id,
                runpod_api_key,
                hf_token,
                aws_access_key_id,
                aws_secret_access_key,
                aws_endpoint_url,
                s3_model_url,
            )
        )
        return

    # Use internal URL for the worker
    # In Cloud Run, we can use the service URL. 
    # If internal_webhook_base_url is set, use that.
    base_url = settings.internal_webhook_base_url
    if not base_url:
        structured_log(
            "ERROR", 
            "internal_webhook_base_url not set; cannot enqueue Cloud Task", 
            deployment_id=deployment_id
        )
        # Fallback? Or fail? Better to fail or fallback.
        # Fallback to async for now to avoid total breakage if config is incomplete
        asyncio.create_task(
            orchestrate_deployment(
                deployment_id,
                runpod_api_key,
                hf_token,
                aws_access_key_id,
                aws_secret_access_key,
                aws_endpoint_url,
                s3_model_url,
            )
        )
        return

    url = f"{base_url}/internal/tasks/orchestrate-deployment"
    payload = {
        "deployment_id": deployment_id,
        # F4: TODO: runpod_api_key and hf_token are passed in plain-text in the Cloud Tasks
        # HTTP request body. This payload is readable in the GCP Cloud Tasks console and logs.
        # Production hardening: store secrets in GCP Secret Manager and pass only a secret
        # reference (e.g. sm://SECRET_NAME), or use Workload Identity + service account
        # scoped to the Secret Manager resource. The secret_cache module already supports
        # sm:// resolution via config.py, so the same pattern could apply here.
        "runpod_api_key": runpod_api_key,
        "hf_token": hf_token,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_endpoint_url": aws_endpoint_url,
        "s3_model_url": s3_model_url,
    }
    
    try:
        client = tasks_v2.CloudTasksClient()
        
        # Construct the request body.
        task = {
            "http_request": {  # Specify the type of request.
                "http_method": tasks_v2.HttpMethod.POST,
                "url": url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(payload).encode(),
            }
        }
        
        # Add internal secret header if configured
        if settings.internal_webhook_secret:
            task["http_request"]["headers"]["X-Visgate-Internal-Secret"] = settings.internal_webhook_secret

        # F5: TODO: Cloud Run targets require OIDC authentication for incoming requests.
        # Without an OIDC token, the task will hit the service unauthenticated. This is
        # currently mitigated by INTERNAL_WEBHOOK_SECRET, but OIDC provides a second
        # layer and is the GCP-recommended approach. To enable:
        #   task["http_request"]["oidc_token"] = {
        #       "service_account_email": settings.cloud_tasks_service_account,
        #       "audience": base_url,
        #   }
        # Requires: CLOUD_TASKS_SERVICE_ACCOUNT env var + SA with roles/run.invoker.
        response = client.create_task(request={"parent": queue_path, "task": task})
        
        structured_log(
            "INFO",
            "Enqueued orchestration task",
            deployment_id=deployment_id,
            metadata={"task_name": response.name},
        )

    except Exception as e:
        structured_log(
            "ERROR",
            f"Failed to enqueue Cloud Task: {e}",
            deployment_id=deployment_id,
            error={"message": str(e)},
        )
        # Fallback to save the day
        asyncio.create_task(
            orchestrate_deployment(
                deployment_id,
                runpod_api_key,
                hf_token,
                aws_access_key_id,
                aws_secret_access_key,
                aws_endpoint_url,
                s3_model_url,
            )
        )
