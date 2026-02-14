"""Google Cloud Tasks integration for background processing."""

import asyncio
import json

from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2

from src.core.config import get_settings
from src.core.logging import structured_log
from src.services.deployment import orchestrate_deployment


async def enqueue_orchestration_task(deployment_id: str) -> None:
    """
    Enqueue a task to orchestrate deployment.
    If cloud_tasks_queue_path is configured, uses Cloud Tasks.
    Otherwise, falls back to asyncio.create_task (not reliable for long ops in Cloud Run).
    """
    settings = get_settings()
    queue_path = settings.cloud_tasks_queue_path

    if not queue_path:
        structured_log(
            "WARNING",
            "Cloud Tasks not confused; falling back to asyncio (unreliable in Cloud Run)",
            deployment_id=deployment_id,
        )
        asyncio.create_task(orchestrate_deployment(deployment_id))
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
        asyncio.create_task(orchestrate_deployment(deployment_id))
        return

    url = f"{base_url}/internal/tasks/orchestrate-deployment"
    payload = {"deployment_id": deployment_id}
    
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

        # Use OIDC token if service account auth is needed (usually required for Cloud Run)
        # For simplicity in this "open source" version, we might assume the queue 
        # has permissions or the service is public/authenticated via secret.
        # But best practice is OIDC.
        # task["http_request"]["oidc_token"] = {"service_account_email": ...} 
        # We will omit OIDC for now as it requires more config (SA email), 
        # and rely on the internal secret + network security.

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
        asyncio.create_task(orchestrate_deployment(deployment_id))
