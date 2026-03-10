"""Google Cloud Tasks integration for background processing."""

import asyncio
import hashlib
import json
import time
from typing import Any

import google.auth
from google.api_core.exceptions import AlreadyExists
from google.cloud import tasks_v2

from src.core.config import get_settings
from src.core.logging import structured_log
from src.services.deployment import orchestrate_deployment


def _runtime_service_account_email() -> str | None:
    """Best-effort resolution of the current ADC service account email."""
    try:
        credentials, _ = google.auth.default()
    except Exception:
        return None
    email = getattr(credentials, "service_account_email", None)
    if not isinstance(email, str) or "@" not in email:
        return None
    return email


def _grant_secret_accessor_bindings(client: Any, secret_path: str) -> None:
    """Grant secret accessor on ephemeral task secrets to service accounts that need it."""
    settings = get_settings()
    service_accounts = {
        sa
        for sa in (
            _runtime_service_account_email(),
            settings.cloud_tasks_service_account,
        )
        if sa
    }
    if not service_accounts:
        return

    role = "roles/secretmanager.secretAccessor"
    policy = client.get_iam_policy(request={"resource": secret_path})
    binding = next((item for item in policy.bindings if item.role == role), None)
    changed = False

    if binding is None:
        binding = policy.bindings.add()
        binding.role = role
        changed = True

    existing_members = set(binding.members)
    for service_account in sorted(service_accounts):
        member = f"serviceAccount:{service_account}"
        if member not in existing_members:
            binding.members.append(member)
            changed = True

    if changed:
        client.set_iam_policy(request={"resource": secret_path, "policy": policy})


def _store_task_secrets(deployment_id: str, project_id: str, secrets: dict[str, Any]) -> str:
    """
    Store per-deployment secrets in Secret Manager and return the secret name.

    Each deployment gets its own secret (name = visgate-dep-{deployment_id}).
    The caller (Cloud Tasks HTTP handler) fetches and destroys the version after use so
    credentials are never left at rest longer than the orchestration window.
    """
    from google.cloud import secretmanager  # lazy import to avoid mandatory GCP client in tests

    secret_id = f"visgate-dep-{deployment_id}"
    parent = f"projects/{project_id}"
    client = secretmanager.SecretManagerServiceClient()

    try:
        client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}},
            }
        )
    except Exception:
        # Secret already exists from a previous attempt; add a new version below.
        pass

    secret_path = f"{parent}/secrets/{secret_id}"
    _grant_secret_accessor_bindings(client, secret_path)
    client.add_secret_version(
        request={"parent": secret_path, "payload": {"data": json.dumps(secrets).encode()}}
    )
    return secret_id


def _cache_task_name(queue_path: str, hf_model_id: str) -> str:
    digest = hashlib.sha256(hf_model_id.encode("utf-8")).hexdigest()[:16]
    # Include 1-hour epoch window so tombstoned tasks (from failed runs) don't
    # block retries after the hour rolls over. The manifest check in the handler
    # is the real dedup guard against redundant uploads.
    window = int(time.time()) // 3600
    return f"{queue_path}/tasks/cache-model-{digest}-{window}"


async def enqueue_orchestration_task(
    deployment_id: str,
    runpod_api_key: str,
    hf_token: str | None,
) -> None:
    """
    Enqueue a task to orchestrate deployment.

    When CLOUD_TASKS_QUEUE_PATH is configured (recommended for production):
      1. Credentials are stored in Secret Manager under visgate-dep-{deployment_id}.
         The Cloud Tasks HTTP body contains only deployment_id + secret_ref — no plain-text
         credentials appear in the GCP console or Cloud Tasks audit logs.
      2. OIDC token is attached when CLOUD_TASKS_SERVICE_ACCOUNT is set, so Cloud Run
         enforces authentication for the internal endpoint.
      3. Cloud Tasks retries on failure, making orchestration durable across Cloud Run
         scale-to-zero events.

    Fallback (no queue configured): asyncio.create_task — works for dev/small scale but
    is not durable. A Cloud Run instance killed mid-orchestration will stall the deployment.
    Set CLOUD_TASKS_QUEUE_PATH to eliminate this risk in production.
    """
    settings = get_settings()
    queue_path = settings.cloud_tasks_queue_path

    if not queue_path:
        structured_log(
            "WARNING",
            "CLOUD_TASKS_QUEUE_PATH not set — using in-process orchestration "
            "(not durable under Cloud Run scale-to-zero)",
            deployment_id=deployment_id,
        )
        asyncio.create_task(
            orchestrate_deployment(
                deployment_id,
                runpod_api_key,
                hf_token,
            )
        )
        return

    base_url = settings.internal_webhook_base_url
    if not base_url:
        structured_log(
            "ERROR",
            "INTERNAL_WEBHOOK_BASE_URL not set; cannot build Cloud Task target URL — "
            "falling back to in-process orchestration",
            deployment_id=deployment_id,
        )
        asyncio.create_task(
            orchestrate_deployment(
                deployment_id,
                runpod_api_key,
                hf_token,
            )
        )
        return

    # Store credentials in Secret Manager. If the runtime lacks IAM mutation rights
    # for ephemeral secrets, fall back to in-process orchestration instead of failing
    # the user-facing deployment create request.
    try:
        secret_ref = _store_task_secrets(
            deployment_id,
            settings.gcp_project_id,
            {
                "runpod_api_key": runpod_api_key,
                "hf_token": hf_token,
            },
        )
    except Exception as e:
        structured_log(
            "ERROR",
            f"Failed to prepare task secrets: {e}; falling back to in-process orchestration",
            deployment_id=deployment_id,
            error={"message": str(e)},
        )
        asyncio.create_task(
            orchestrate_deployment(
                deployment_id,
                runpod_api_key,
                hf_token,
            )
        )
        return

    url = f"{base_url}/internal/tasks/orchestrate-deployment"
    # No credentials in the body — only the deployment_id and Secret Manager reference.
    payload = {"deployment_id": deployment_id, "secret_ref": secret_ref}

    http_request: dict[str, Any] = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": url,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload).encode(),
    }

    if settings.internal_webhook_secret:
        http_request["headers"]["X-Visgate-Internal-Secret"] = settings.internal_webhook_secret

    # OIDC token gives Cloud Run a verified caller identity on top of the shared secret.
    if settings.cloud_tasks_service_account:
        http_request["oidc_token"] = {
            "service_account_email": settings.cloud_tasks_service_account,
            "audience": base_url,
        }

    try:
        client = tasks_v2.CloudTasksClient()
        response = client.create_task(request={"parent": queue_path, "task": {"http_request": http_request}})
        structured_log(
            "INFO",
            "Enqueued orchestration task via Cloud Tasks",
            deployment_id=deployment_id,
            metadata={"task_name": response.name},
        )

    except Exception as e:
        structured_log(
            "ERROR",
            f"Failed to enqueue Cloud Task: {e}; falling back to in-process orchestration",
            deployment_id=deployment_id,
            error={"message": str(e)},
        )
        asyncio.create_task(
            orchestrate_deployment(
                deployment_id,
                runpod_api_key,
                hf_token,
            )
        )


async def enqueue_cache_model_task(
    hf_model_id: str,
    hf_token: str | None = None,
) -> None:
    """
    Enqueue a background task to download HF model and cache it to R2.
    Runs independently of deployment orchestration.
    """
    settings = get_settings()
    queue_path = settings.cloud_tasks_queue_path

    if not queue_path or not settings.r2_access_key_id_rw:
        structured_log(
            "WARNING",
            "Cache modeling skipped: no Cloud Tasks queue or R2 RW credentials configured",
            hf_model_id=hf_model_id,
        )
        return

    base_url = settings.internal_webhook_base_url
    if not base_url:
        structured_log(
            "WARNING",
            "Cache modeling skipped: INTERNAL_WEBHOOK_BASE_URL not set",
            hf_model_id=hf_model_id,
        )
        return

    url = f"{base_url}/internal/tasks/cache-model"
    payload = {"hf_model_id": hf_model_id, "hf_token": hf_token or ""}

    http_request: dict[str, Any] = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": url,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload).encode(),
    }

    if settings.internal_webhook_secret:
        http_request["headers"]["X-Visgate-Internal-Secret"] = settings.internal_webhook_secret

    if settings.cloud_tasks_service_account:
        http_request["oidc_token"] = {
            "service_account_email": settings.cloud_tasks_service_account,
            "audience": base_url,
        }

    try:
        client = tasks_v2.CloudTasksClient()
        response = client.create_task(
            request={
                "parent": queue_path,
                "task": {
                    "name": _cache_task_name(queue_path, hf_model_id),
                    "http_request": http_request,
                },
            }
        )
        structured_log(
            "INFO",
            "Enqueued cache modeling task",
            hf_model_id=hf_model_id,
            metadata={"task_name": response.name},
        )
    except AlreadyExists:
        structured_log(
            "INFO",
            "Cache modeling task already queued or recently executed",
            hf_model_id=hf_model_id,
            operation="r2.cache_model.enqueue",
        )
    except Exception as e:
        structured_log(
            "WARNING",
            f"Failed to enqueue cache modeling task: {e}",
            hf_model_id=hf_model_id,
            error={"message": str(e)},
        )
