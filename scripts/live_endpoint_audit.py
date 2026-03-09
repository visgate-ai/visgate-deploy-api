#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import requests


API_BASE = os.environ.get("API_BASE", "https://visgate-deploy-api-wxup7pxrsa-ey.a.run.app").rstrip("/")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "visgate")
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD") or ""
HF_TOKEN = os.environ.get("HF") or os.environ.get("HF_TOKEN") or ""
DEPLOY_TIMEOUT_SECONDS = int(os.environ.get("DEPLOY_TIMEOUT_SECONDS", "1800"))
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "900"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "8"))
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "artifacts/live-audit"))


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def bool_summary(ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": ok, "details": details or {}}


def shell_secret(name: str) -> str:
    cmd = [
        "gcloud",
        "secrets",
        "versions",
        "access",
        "latest",
        f"--secret={name}",
        f"--project={PROJECT_ID}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def headers() -> dict[str, str]:
    if not RUNPOD_KEY:
        return {}
    base = {
        "Authorization": f"Bearer {RUNPOD_KEY}",
        "X-Runpod-Api-Key": RUNPOD_KEY,
    }
    if HF_TOKEN:
        base["X-HF-Api-Key"] = HF_TOKEN
    return base


def json_request(method: str, path: str, *, expected: int | None = None, timeout: int = 60, **kwargs: Any) -> requests.Response:
    response = requests.request(
        method,
        f"{API_BASE}{path}",
        headers={**headers(), **kwargs.pop("headers", {})},
        timeout=timeout,
        **kwargs,
    )
    if expected is not None and response.status_code != expected:
        raise RuntimeError(f"{method} {path} expected {expected}, got {response.status_code}: {response.text[:500]}")
    return response


def create_webhook_token() -> tuple[str, str]:
    response = requests.post(
        "https://webhook.site/token",
        json={},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload["uuid"]
    return f"https://webhook.site/{token}", token


def fetch_webhook_requests(token: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"https://webhook.site/token/{token}/requests?sorting=newest&per_page=10",
        headers={"Accept": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("data") or []


def wait_for_webhook(token: str, predicate, timeout_seconds: int) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for item in fetch_webhook_requests(token):
            content = item.get("content") or ""
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                continue
            if predicate(payload):
                return payload
        time.sleep(5)
    return None


def validate_schema_fields(payload: dict[str, Any], required_fields: list[str]) -> dict[str, Any]:
    missing = [field for field in required_fields if field not in payload]
    return bool_summary(not missing, {"missing": missing, "keys": sorted(payload.keys())})


def read_sse_event(path: str, *, timeout_seconds: int = 25) -> dict[str, Any]:
    with requests.get(f"{API_BASE}{path}", headers=headers(), stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        event_name = None
        data_lines: list[str] = []
        started = time.monotonic()
        for raw_line in response.iter_lines(decode_unicode=True):
            if time.monotonic() - started > timeout_seconds:
                break
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                if event_name or data_lines:
                    data = json.loads("\n".join(data_lines)) if data_lines else None
                    return {"event": event_name, "data": data}
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
    raise RuntimeError(f"No SSE event received from {path} within {timeout_seconds}s")


def output_s3_config() -> dict[str, str]:
    access_id = os.environ.get("OUTPUT_ACCESS_ID") or shell_secret("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_RW")
    access_secret = os.environ.get("OUTPUT_ACCESS_SECRET") or shell_secret("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_RW")
    bucket_name = os.environ.get("OUTPUT_BUCKET_NAME") or shell_secret("VISGATE_DEPLOY_API_INFERENCE_R2_BUCKET_NAME")
    endpoint_url = os.environ.get("OUTPUT_ENDPOINT_URL") or shell_secret("VISGATE_DEPLOY_API_S3_API_R2")
    return {
        "accessId": access_id,
        "accessSecret": access_secret,
        "bucketName": bucket_name,
        "endpointUrl": endpoint_url,
        "keyPrefix": f"live-audit/{int(time.time())}",
    }


def head_artifact(artifact: dict[str, Any], s3_config: dict[str, str]) -> dict[str, Any]:
    key = artifact.get("key")
    bucket_name = artifact.get("bucket_name")
    if not key or not bucket_name:
        return bool_summary(False, {"reason": "artifact missing key or bucket_name", "artifact": artifact})

    client = boto3.client(
        "s3",
        endpoint_url=s3_config["endpointUrl"],
        aws_access_key_id=s3_config["accessId"],
        aws_secret_access_key=s3_config["accessSecret"],
        region_name="auto",
    )
    meta = client.head_object(Bucket=bucket_name, Key=key)
    return bool_summary(True, {"content_length": meta.get("ContentLength"), "content_type": meta.get("ContentType")})


def poll_json(path: str, *, terminal_statuses: set[str], timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = json_request("GET", path, expected=200, timeout=60)
        last_payload = response.json()
        if last_payload.get("status") in terminal_statuses:
            return last_payload
        time.sleep(POLL_SECONDS)
    raise RuntimeError(f"Timed out polling {path}; last payload={json.dumps(last_payload)[:1000]}")


def metrics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta = {
        "deployments_created_total": after.get("deployments_created_total", 0) - before.get("deployments_created_total", 0),
        "webhook_delivery_failures_total": after.get("webhook_delivery_failures_total", 0) - before.get("webhook_delivery_failures_total", 0),
        "runpod_api_errors_total": after.get("runpod_api_errors_total", 0) - before.get("runpod_api_errors_total", 0),
        "deployments_ready_duration_count": (
            after.get("deployments_ready_duration_seconds", {}).get("count", 0)
            - before.get("deployments_ready_duration_seconds", {}).get("count", 0)
        ),
    }
    return delta


def main() -> int:
    if not RUNPOD_KEY:
        raise SystemExit("Set RUNPOD_API_KEY or RUNPOD")

    internal_secret = os.environ.get("INTERNAL_WEBHOOK_SECRET") or shell_secret("VISGATE_DEPLOY_API_INTERNAL_WEBHOOK_SECRET")
    s3_config = output_s3_config()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "api_base": API_BASE,
        "started_at": now_iso(),
        "checks": {},
        "artifacts": {},
        "resources": {},
    }

    deployment_id = None
    inference_job_id = None
    cancel_job_id = None

    metrics_before = json_request("GET", "/metrics", expected=200).json()
    report["checks"]["root"] = bool_summary(
        (root := json_request("GET", "/", expected=200).json()).get("service") == "visgate-deploy-api",
        root,
    )
    report["checks"]["health"] = bool_summary(
        (health := json_request("GET", "/health", expected=200).json()).get("status") == "ok",
        health,
    )
    readiness_resp = json_request("GET", "/readiness", expected=200).json()
    report["checks"]["readiness"] = bool_summary(readiness_resp.get("status") == "ready", readiness_resp)
    report["checks"]["metrics_before"] = bool_summary(True, metrics_before)

    models = json_request("GET", "/v1/models", expected=200).json()
    report["checks"]["models_list"] = bool_summary(
        models.get("total", 0) > 0 and any(m.get("model_id") == "stabilityai/sdxl-turbo" for m in models.get("models", [])),
        {"total": models.get("total"), "first_models": models.get("models", [])[:5]},
    )

    search = json_request("GET", "/v1/models/search", expected=200, params={"q": "sdxl", "task": "text-to-image", "limit": 5}).json()
    report["checks"]["models_search"] = bool_summary(
        len(search.get("results", [])) > 0,
        {"query": search.get("query"), "results": search.get("results", [])[:3]},
    )

    provider_validate = json_request(
        "POST",
        "/v1/providers/validate",
        expected=200,
        json={"provider": "runpod", "api_key": RUNPOD_KEY},
    ).json()
    report["checks"]["provider_validate_runpod"] = bool_summary(provider_validate.get("valid") is True, provider_validate)

    provider_unsupported = json_request(
        "POST",
        "/v1/providers/validate",
        expected=200,
        json={"provider": "unknown", "api_key": "x"},
    ).json()
    report["checks"]["provider_validate_unsupported"] = bool_summary(
        provider_unsupported.get("valid") is False and "Unsupported provider" in provider_unsupported.get("message", ""),
        provider_unsupported,
    )

    gpus = json_request("GET", "/v1/deployments/gpus", expected=200).json()
    report["checks"]["deployments_gpus"] = bool_summary(len(gpus.get("gpus", [])) > 0, {"sample": gpus.get("gpus", [])[:3]})

    deployments_before = json_request("GET", "/v1/deployments", expected=200).json()
    report["checks"]["deployments_list_before"] = bool_summary(True, {"total": deployments_before.get("total")})

    deployment_webhook_url, deployment_webhook_token = create_webhook_token()
    deployment_create = json_request(
        "POST",
        "/v1/deployments",
        expected=202,
        json={
            "hf_model_id": "stabilityai/sdxl-turbo",
            "task": "text_to_image",
            "cache_scope": "shared",
            "user_webhook_url": deployment_webhook_url,
        },
        timeout=90,
    ).json()
    deployment_id = deployment_create["deployment_id"]
    report["resources"]["deployment_id"] = deployment_id
    report["checks"]["deployment_create"] = validate_schema_fields(
        deployment_create,
        [
            "deployment_id",
            "status",
            "model_id",
            "estimated_ready_seconds",
            "estimated_ready_at",
            "poll_interval_seconds",
            "stream_url",
            "webhook_url",
            "path",
            "created_at",
        ],
    )

    report["checks"]["deployment_stream"] = bool_summary(True, read_sse_event(f"/v1/deployments/{deployment_id}/stream"))
    get_deployment_initial = json_request("GET", f"/v1/deployments/{deployment_id}", expected=200).json()
    report["checks"]["deployment_get_initial"] = validate_schema_fields(
        get_deployment_initial,
        ["deployment_id", "status", "hf_model_id", "logs", "created_at"],
    )
    cost_initial = json_request("GET", f"/v1/deployments/{deployment_id}/cost", expected=200).json()
    report["checks"]["deployment_cost_initial"] = validate_schema_fields(cost_initial, ["deployment_id", "status"])

    unauthorized_internal = requests.post(f"{API_BASE}/internal/logs/{deployment_id}", json={"level": "INFO", "message": "unauthorized"}, timeout=30)
    report["checks"]["internal_log_unauthorized"] = bool_summary(
        unauthorized_internal.status_code == 403,
        {"status_code": unauthorized_internal.status_code, "body": unauthorized_internal.text[:300]},
    )

    authorized_log = json_request(
        "POST",
        f"/internal/logs/{deployment_id}",
        expected=200,
        headers={"X-Visgate-Internal-Secret": internal_secret},
        json={"level": "INFO", "message": "live-endpoint-audit"},
    ).json()
    report["checks"]["internal_log_authorized"] = bool_summary(authorized_log.get("status") == "ok", authorized_log)
    report["checks"]["deployment_logs_stream"] = bool_summary(True, read_sse_event(f"/v1/deployments/{deployment_id}/logs/stream"))

    deployment_final = poll_json(
        f"/v1/deployments/{deployment_id}",
        terminal_statuses={"ready", "failed", "webhook_failed", "deleted"},
        timeout_seconds=DEPLOY_TIMEOUT_SECONDS,
    )
    report["checks"]["deployment_ready"] = bool_summary(
        deployment_final.get("status") == "ready",
        {
            "status": deployment_final.get("status"),
            "error": deployment_final.get("error"),
            "endpoint_url": deployment_final.get("endpoint_url"),
            "gpu_allocated": deployment_final.get("gpu_allocated"),
        },
    )
    report["checks"]["deployment_persisted_worker_fields"] = bool_summary(
        bool(deployment_final.get("runpod_endpoint_id")) and bool(deployment_final.get("endpoint_url")),
        {
            "runpod_endpoint_id": deployment_final.get("runpod_endpoint_id"),
            "endpoint_url": deployment_final.get("endpoint_url"),
            "logs_tail": deployment_final.get("logs", [])[-5:],
        },
    )

    deployment_webhook = wait_for_webhook(
        deployment_webhook_token,
        lambda payload: payload.get("deployment_id") == deployment_id and payload.get("status") == "ready",
        timeout_seconds=120,
    )
    report["artifacts"]["deployment_webhook_payload"] = deployment_webhook
    report["checks"]["deployment_webhook_payload"] = bool_summary(
        bool(deployment_webhook)
        and deployment_webhook.get("event") == "deployment_ready"
        and deployment_webhook.get("deployment_id") == deployment_id
        and deployment_webhook.get("endpoint_url") == deployment_final.get("endpoint_url")
        and isinstance(deployment_webhook.get("duration_seconds"), (int, float))
        and deployment_webhook.get("duration_seconds", -1) >= 0
        and deployment_webhook.get("usage_example", {}).get("method") == "POST"
        and deployment_webhook.get("usage_example", {}).get("url") == deployment_final.get("endpoint_url"),
        deployment_webhook or {"missing": True},
    )

    deployments_after_create = json_request("GET", "/v1/deployments", expected=200).json()
    report["checks"]["deployments_list_after_create"] = bool_summary(
        any(item.get("deployment_id") == deployment_id for item in deployments_after_create.get("deployments", [])),
        {"total": deployments_after_create.get("total")},
    )

    inference_webhook_url, inference_webhook_token = create_webhook_token()
    inference_create = json_request(
        "POST",
        "/v1/inference/jobs",
        expected=202,
        json={
            "deployment_id": deployment_id,
            "input": {
                "prompt": "A bright red apple on a white table, product photo",
                "num_inference_steps": 4,
                "guidance_scale": 0.0,
                "width": 512,
                "height": 512,
            },
            "user_webhook_url": inference_webhook_url,
            "s3Config": s3_config,
        },
        timeout=90,
    ).json()
    inference_job_id = inference_create["job_id"]
    report["resources"]["inference_job_id"] = inference_job_id
    report["checks"]["inference_create"] = validate_schema_fields(
        inference_create,
        ["job_id", "deployment_id", "provider", "provider_job_id", "status", "provider_status", "output_destination", "created_at"],
    )

    job_list = json_request("GET", "/v1/inference/jobs", expected=200).json()
    report["checks"]["inference_list"] = bool_summary(
        any(job.get("job_id") == inference_job_id for job in job_list.get("jobs", [])),
        {"total": job_list.get("total")},
    )

    job_final = poll_json(
        f"/v1/inference/jobs/{inference_job_id}",
        terminal_statuses={"completed", "failed", "cancelled", "expired"},
        timeout_seconds=JOB_TIMEOUT_SECONDS,
    )
    report["artifacts"]["inference_job_final"] = job_final
    metrics = job_final.get("metrics") or {}
    report["checks"]["inference_complete"] = bool_summary(
        job_final.get("status") == "completed",
        {"status": job_final.get("status"), "error": job_final.get("error"), "metrics": metrics},
    )
    report["checks"]["inference_metrics"] = bool_summary(
        isinstance(metrics.get("wall_clock_ms"), int)
        and metrics.get("wall_clock_ms", -1) >= 0
        and (metrics.get("queue_ms") is None or metrics.get("queue_ms") >= 0)
        and (metrics.get("execution_ms") is None or metrics.get("execution_ms") >= 0),
        metrics,
    )
    report["checks"]["inference_artifact"] = head_artifact(job_final.get("artifact") or {}, s3_config)

    inference_webhook = wait_for_webhook(
        inference_webhook_token,
        lambda payload: payload.get("job_id") == inference_job_id,
        timeout_seconds=120,
    )
    report["artifacts"]["inference_webhook_payload"] = inference_webhook
    report["checks"]["inference_webhook_payload"] = bool_summary(
        bool(inference_webhook)
        and inference_webhook.get("event") == "inference_job_completed"
        and inference_webhook.get("job_id") == inference_job_id
        and isinstance((inference_webhook.get("metrics") or {}).get("wall_clock_ms"), int)
        and bool((inference_webhook.get("artifact") or {}).get("key")),
        inference_webhook or {"missing": True},
    )

    cancel_create = json_request(
        "POST",
        "/v1/inference/jobs",
        expected=202,
        json={
            "deployment_id": deployment_id,
            "input": {
                "prompt": "A metallic blue bicycle studio photo",
                "num_inference_steps": 20,
                "guidance_scale": 0.0,
                "width": 512,
                "height": 512,
            },
            "s3Config": s3_config,
        },
        timeout=90,
    ).json()
    cancel_job_id = cancel_create["job_id"]
    report["resources"]["cancel_job_id"] = cancel_job_id

    cancel_response = json_request("POST", f"/v1/inference/jobs/{cancel_job_id}/cancel", expected=200, timeout=120).json()
    report["checks"]["inference_cancel"] = bool_summary(
        cancel_response.get("job_id") == cancel_job_id and cancel_response.get("status") in {"cancelled", "failed", "running", "queued"},
        cancel_response,
    )

    retry_response = json_request("POST", f"/v1/inference/jobs/{cancel_job_id}/retry", expected=200, timeout=120).json()
    report["checks"]["inference_retry"] = bool_summary(
        retry_response.get("job_id") == cancel_job_id and retry_response.get("status") in {"queued", "running", "completed", "cancelled"},
        retry_response,
    )

    metrics_after = json_request("GET", "/metrics", expected=200).json()
    delta = metrics_delta(metrics_before, metrics_after)
    report["checks"]["metrics_after"] = bool_summary(True, metrics_after)
    report["checks"]["metrics_delta"] = bool_summary(
        delta["deployments_created_total"] >= 1 and delta["deployments_ready_duration_count"] >= 1,
        delta,
    )

    delete_response = json_request("DELETE", f"/v1/deployments/{deployment_id}", expected=204, timeout=120)
    report["checks"]["deployment_delete"] = bool_summary(delete_response.status_code == 204, {"status_code": delete_response.status_code})
    deleted_state = json_request("GET", f"/v1/deployments/{deployment_id}", expected=200).json()
    report["checks"]["deployment_deleted_state"] = bool_summary(deleted_state.get("status") == "deleted", deleted_state)

    report["finished_at"] = now_iso()
    report["overall_ok"] = all(check.get("ok") for check in report["checks"].values())

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / f"live-endpoint-audit-{timestamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(report_path)
    print(json.dumps({"overall_ok": report["overall_ok"], "checks": {k: v["ok"] for k, v in report["checks"].items()}}, ensure_ascii=False, indent=2))
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())