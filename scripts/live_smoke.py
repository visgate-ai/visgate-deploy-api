#!/usr/bin/env python3
"""Run live image/audio/video smoke tests against the hosted Deploy API."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socketserver
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import boto3

API_BASE = os.environ["API_BASE"].rstrip("/")
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
INTERNAL_WEBHOOK_SECRET = os.environ.get("INTERNAL_WEBHOOK_SECRET", "")
R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"]

INPUT_R2_CLIENT = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=os.environ["INPUT_R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["INPUT_R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)
OUTPUT_R2_CLIENT = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=os.environ["OUTPUT_R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["OUTPUT_R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)
INPUT_BUCKET = os.environ["INPUT_R2_BUCKET_NAME"]
OUTPUT_BUCKET = os.environ["OUTPUT_R2_BUCKET_NAME"]

DEPLOYMENT_TIMEOUTS = {
    "image": 1800,
    "audio": 1500,
    "video": 3600,
}

JOB_TIMEOUTS = {
    "image": 1200,
    "audio": 900,
    "video": 2400,
}

CACHE_TIMEOUTS = {
    "image": 2400,
    "audio": 1800,
    "video": 5400,
}

MODALITY_CONFIGS: dict[str, dict[str, Any]] = {
    "image": {
        "deployment": {
            "hf_model_id": "stabilityai/sd-turbo",
            "task": "image_to_image",
        },
        "job": {
            "task": "image_to_image",
            "input": {
                "prompt": "convert the input into a crisp product photo on white background",
                "num_inference_steps": 2,
                "input_image_url": "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png",
            },
        },
        "staged_field": "input_image_r2",
        "expect_output": True,
    },
    "audio": {
        "deployment": {
            "hf_model_id": "openai/whisper-large-v3-turbo",
            "task": "speech_to_text",
        },
        "job": {
            "task": "speech_to_text",
            "input": {
                "audio_url": "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_george_0.wav",
            },
        },
        "staged_field": "audio_r2",
        "expect_output": False,
    },
    "video": {
        "deployment": {
            "hf_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
            "task": "text_to_video",
        },
        "job": {
            "task": "text_to_video",
            "input": {
                "prompt": "a red cube slowly rotating on a white table, studio lighting",
                "num_inference_steps": 4,
                "num_frames": 8,
                "fps": 8,
            },
        },
        "staged_field": None,
        "expect_output": True,
    },
}

SMOKE_MODEL_IDS = {config["deployment"]["hf_model_id"] for config in MODALITY_CONFIGS.values()}


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class _WebhookHandler(BaseHTTPRequestHandler):
    server: "WebhookServer"

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"raw": raw_body.decode("utf-8", errors="replace")}
        self.server.record_event(self.path, payload, dict(self.headers.items()))
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


class WebhookServer(_ReusableThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]):
        super().__init__(server_address, _WebhookHandler)
        self.events: dict[str, list[dict[str, Any]]] = {}
        self._condition = threading.Condition()

    def record_event(self, path: str, payload: dict[str, Any], headers: dict[str, str]) -> None:
        with self._condition:
            self.events.setdefault(path, []).append(
                {
                    "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "payload": payload,
                    "headers": headers,
                }
            )
            self._condition.notify_all()

    def wait_for_event(self, path: str, *, timeout_seconds: int) -> dict[str, Any] | None:
        deadline = time.time() + timeout_seconds
        with self._condition:
            while time.time() < deadline:
                events = self.events.get(path)
                if events:
                    return events[0]
                remaining = deadline - time.time()
                self._condition.wait(timeout=max(0.1, min(5.0, remaining)))
        return None


@dataclass
class WebhookCapture:
    server: WebhookServer
    thread: threading.Thread
    base_url: str
    tunnel_process: subprocess.Popen[str] | None = None


def _find_free_port() -> int:
    with contextlib.closing(socketserver.socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tunnel_url(process: subprocess.Popen[str], *, timeout_seconds: int = 60) -> str:
    assert process.stderr is not None
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        line = process.stderr.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        marker = "trycloudflare.com"
        if marker in line:
            for token in line.split():
                if token.startswith("https://") and marker in token:
                    return token.rstrip()
    stderr_tail = ""
    if process.stderr is not None:
        try:
            stderr_tail = process.stderr.read()
        except Exception:
            stderr_tail = ""
    raise RuntimeError(f"Timed out waiting for cloudflared tunnel URL; stderr={stderr_tail[-500:]}")


def _start_webhook_capture() -> WebhookCapture:
    configured_base = os.environ.get("WEBHOOK_BASE_URL", "").strip().rstrip("/")
    port = int(os.environ.get("WEBHOOK_PORT", "0")) or _find_free_port()
    server = WebhookServer(("127.0.0.1", port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if configured_base:
        return WebhookCapture(server=server, thread=thread, base_url=configured_base)

    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = _wait_for_tunnel_url(process)
    return WebhookCapture(server=server, thread=thread, base_url=base_url.rstrip("/"), tunnel_process=process)


def _stop_webhook_capture(capture: WebhookCapture) -> None:
    capture.server.shutdown()
    capture.server.server_close()
    capture.thread.join(timeout=5)
    process = capture.tunnel_process
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, method=method, data=data, headers=headers or _auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def _request_no_content(method: str, url: str) -> None:
    request = urllib.request.Request(url, method=method, headers=_auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=120):
            return
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def _poll_json(url: str, terminal_statuses: set[str], *, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        last_payload = _request_json("GET", url)
        status = str(last_payload.get("status", ""))
        if status in terminal_statuses:
            return last_payload
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for terminal status at {url}; last payload={last_payload}")


@dataclass
class SseCapture:
    process: subprocess.Popen[str]
    events: list[dict[str, Any]]
    thread: threading.Thread


def _start_sse_capture(url: str) -> SseCapture:
    process = subprocess.Popen(
        ["curl", "-NsS", "-H", f"Authorization: Bearer {RUNPOD_API_KEY}", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    events: list[dict[str, Any]] = []

    def _reader() -> None:
        current_event = "message"
        current_data: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip("\n")
            if stripped.startswith("event: "):
                current_event = stripped.split(": ", 1)[1]
            elif stripped.startswith("data: "):
                current_data.append(stripped.split(": ", 1)[1])
            elif stripped == "":
                if current_data:
                    data_text = "\n".join(current_data)
                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        data = {"raw": data_text}
                    events.append({"event": current_event, "data": data})
                current_event = "message"
                current_data = []

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return SseCapture(process=process, events=events, thread=thread)


def _stop_sse_capture(capture: SseCapture) -> list[dict[str, Any]]:
    if capture.process.poll() is None:
        capture.process.terminate()
        try:
            capture.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            capture.process.kill()
    capture.thread.join(timeout=5)
    return capture.events


def _webhook_path(kind: str, modality: str) -> str:
    return f"/{kind}-{modality}"


def _webhook_url(base_url: str, kind: str, modality: str) -> str:
    return f"{base_url}{_webhook_path(kind, modality)}"


def _head_object(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _list_prefix(client: Any, bucket: str, prefix: str) -> list[str]:
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception:
        return []
    return [item["Key"] for item in response.get("Contents", [])]


def _cloud_log_seen(deployment_id: str, started_at_iso: str) -> bool:
    if not GCP_PROJECT_ID:
        return False
    query = (
        'resource.type="cloud_run_revision" '
        'AND resource.labels.service_name="visgate-deploy-api" '
        f'AND httpRequest.requestUrl:"/internal/logs/{deployment_id}" '
        f'AND timestamp>="{started_at_iso}"'
    )
    command = [
        "gcloud",
        "logging",
        "read",
        query,
        "--project",
        GCP_PROJECT_ID,
        "--limit",
        "1",
        "--format=value(httpRequest.status)",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def _webhook_delivery_seen(webhook_url: str, started_at_iso: str, *, timeout_seconds: int = 180) -> bool:
    if not GCP_PROJECT_ID:
        return False
    deadline = time.time() + timeout_seconds
    query = (
        'resource.type="cloud_run_revision" '
        'AND resource.labels.service_name="visgate-deploy-api" '
        'AND jsonPayload.message="Webhook delivered successfully" '
        'AND jsonPayload.operation="webhook.notify" '
        f'AND jsonPayload.metadata.url="{webhook_url}" '
        f'AND timestamp>="{started_at_iso}"'
    )
    command = [
        "gcloud",
        "logging",
        "read",
        query,
        "--project",
        GCP_PROJECT_ID,
        "--limit",
        "1",
        "--format=value(timestamp)",
    ]
    while time.time() < deadline:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.stdout.strip():
            return True
        time.sleep(5)
    return False


def _wait_for_webhook_event(
    capture: WebhookCapture,
    *,
    kind: str,
    modality: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    path = _webhook_path(kind, modality)
    event = capture.server.wait_for_event(path, timeout_seconds=timeout_seconds)
    if event:
        return event
    webhook_url = _webhook_url(capture.base_url, kind, modality)
    if _webhook_delivery_seen(webhook_url, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - timeout_seconds)), timeout_seconds=30):
        return {"received_via": "cloud_log_only", "payload": None, "headers": {}}
    raise RuntimeError(f"Timed out waiting for {kind} webhook delivery for {webhook_url}")


def _model_catalog() -> list[dict[str, Any]]:
    response = _request_json("GET", f"{API_BASE}/v1/models", headers={"Content-Type": "application/json"})
    return list(response.get("models") or [])


def _model_cached(model_id: str) -> bool:
    for item in _model_catalog():
        if item.get("model_id") == model_id:
            return bool(item.get("cached"))
    return False


def _cache_model(model_id: str, modality: str) -> dict[str, Any] | None:
    if not INTERNAL_WEBHOOK_SECRET:
        return None
    return _request_json(
        "POST",
        f"{API_BASE}/internal/tasks/cache-model",
        {"hf_model_id": model_id, "hf_token": HF_TOKEN},
        headers={
            "Content-Type": "application/json",
            "X-Visgate-Internal-Secret": INTERNAL_WEBHOOK_SECRET,
        },
        timeout_seconds=CACHE_TIMEOUTS[modality],
    )


def _ensure_model_cached(modality: str) -> dict[str, Any]:
    model_id = str(MODALITY_CONFIGS[modality]["deployment"]["hf_model_id"])
    result: dict[str, Any] = {
        "model_id": model_id,
        "cached_before": _model_cached(model_id),
    }
    if result["cached_before"]:
        result["cache_status"] = "already_cached"
        result["cached_after"] = True
        return result
    cache_response = _cache_model(model_id, modality)
    if cache_response is not None:
        result["cache_response"] = cache_response
        result["cache_status"] = str(cache_response.get("status", "unknown"))
    else:
        result["cache_status"] = "skipped"

    deadline = time.time() + 180
    while time.time() < deadline:
        if _model_cached(model_id):
            result["cached_after"] = True
            return result
        time.sleep(5)

    result["cached_after"] = _model_cached(model_id)
    if not result["cached_after"] and cache_response is not None:
        raise RuntimeError(f"{modality} model did not appear as cached after internal cache task: {model_id}")
    return result


def _prepare_models(modalities: list[str]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for modality in modalities:
        model_id = str(MODALITY_CONFIGS[modality]["deployment"]["hf_model_id"])
        if model_id in seen_models:
            continue
        prepared.append({"modality": modality, **_ensure_model_cached(modality)})
        seen_models.add(model_id)
    return prepared


def _list_deployments(limit: int = 100) -> list[dict[str, Any]]:
    response = _request_json("GET", f"{API_BASE}/v1/deployments?limit={limit}")
    return list(response.get("deployments") or [])


def _delete_deployment(deployment_id: str) -> None:
    _request_no_content("DELETE", f"{API_BASE}/v1/deployments/{deployment_id}")


def _cleanup_stale_smoke_deployments() -> None:
    deployments = _list_deployments()
    deleted_any = False
    for deployment in deployments:
        deployment_id = deployment.get("deployment_id")
        hf_model_id = deployment.get("hf_model_id")
        status = deployment.get("status")
        if not deployment_id or hf_model_id not in SMOKE_MODEL_IDS:
            continue
        if status == "deleted":
            continue
        _delete_deployment(deployment_id)
        deleted_any = True
    if deleted_any:
        time.sleep(15)


def _deployment_and_job(modality: str) -> dict[str, Any]:
    config = MODALITY_CONFIGS[modality]
    webhook_capture = _start_webhook_capture()
    deployment_webhook_url = _webhook_url(webhook_capture.base_url, "deployment", modality)
    job_webhook_url = _webhook_url(webhook_capture.base_url, "job", modality)
    deployment_payload = dict(config["deployment"])
    deployment_payload["hf_token"] = HF_TOKEN
    deployment_payload["user_runpod_key"] = RUNPOD_API_KEY
    deployment_payload["user_webhook_url"] = deployment_webhook_url

    deployment_response = _request_json("POST", f"{API_BASE}/v1/deployments", deployment_payload)
    deployment_id = deployment_response["deployment_id"]
    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        status_capture = _start_sse_capture(f"{API_BASE}/v1/deployments/{deployment_id}/stream")
        logs_capture = _start_sse_capture(f"{API_BASE}/v1/deployments/{deployment_id}/logs/stream")
        deployment_final = _poll_json(
            f"{API_BASE}/v1/deployments/{deployment_id}",
            {"ready", "failed", "webhook_failed", "deleted"},
            timeout_seconds=DEPLOYMENT_TIMEOUTS[modality],
        )
        status_events = _stop_sse_capture(status_capture)
        log_events = _stop_sse_capture(logs_capture)

        result: dict[str, Any] = {
            "modality": modality,
            "deployment_id": deployment_id,
            "deployment": deployment_final,
            "deployment_status_events": status_events,
            "deployment_log_events": log_events,
            "live_log_tunnel_seen": _cloud_log_seen(deployment_id, started_at_iso),
        }

        if deployment_final["status"] != "ready":
            raise RuntimeError(f"{modality} deployment did not become ready: {deployment_final}")

        result["deployment_webhook_url"] = deployment_webhook_url
        deployment_webhook_event = _wait_for_webhook_event(
            webhook_capture,
            kind="deployment",
            modality=modality,
            timeout_seconds=300,
        )
        result["deployment_webhook_verified"] = True
        result["deployment_webhook_event"] = deployment_webhook_event

        job_payload = dict(config["job"])
        job_payload["deployment_id"] = deployment_id
        job_payload["user_webhook_url"] = job_webhook_url
        job_response = _request_json("POST", f"{API_BASE}/v1/inference/jobs", job_payload)
        job_id = job_response["job_id"]
        job_final = _poll_json(
            f"{API_BASE}/v1/inference/jobs/{job_id}",
            {"completed", "failed", "cancelled", "expired"},
            timeout_seconds=JOB_TIMEOUTS[modality],
        )

        result["job_id"] = job_id
        result["job"] = job_final
        result["job_webhook_url"] = job_webhook_url
        job_webhook_event = _wait_for_webhook_event(
            webhook_capture,
            kind="job",
            modality=modality,
            timeout_seconds=300,
        )
        result["job_webhook_verified"] = True
        result["job_webhook_event"] = job_webhook_event

        staged_field = config.get("staged_field")
        if staged_field:
            staged = (job_final.get("input") or {}).get(staged_field)
            if not isinstance(staged, dict) or not staged.get("key"):
                raise RuntimeError(f"{modality} job missing staged input metadata: {job_final.get('input')}")
            result["staged_input"] = staged
            result["staged_input_verified"] = _head_object(INPUT_R2_CLIENT, INPUT_BUCKET, staged["key"])
            if not result["staged_input_verified"]:
                raise RuntimeError(f"{modality} staged input object not found in R2 input bucket: {staged['key']}")

        if config.get("expect_output"):
            artifact = job_final.get("artifact")
            if not isinstance(artifact, dict):
                raise RuntimeError(f"{modality} job missing artifact metadata: {job_final}")
            result["artifact"] = artifact
            artifact_key = artifact.get("key")
            if artifact_key:
                result["output_verified"] = _head_object(OUTPUT_R2_CLIENT, OUTPUT_BUCKET, artifact_key)
            else:
                prefix = ((job_final.get("output_destination") or {}).get("key_prefix") or "").strip("/")
                listed = _list_prefix(OUTPUT_R2_CLIENT, OUTPUT_BUCKET, prefix) if prefix else []
                result["output_listed_keys"] = listed
                result["output_verified"] = bool(listed)
            if not result["output_verified"]:
                raise RuntimeError(f"{modality} output object could not be verified in R2 output bucket")

        return result
    finally:
        _stop_webhook_capture(webhook_capture)
        try:
            _delete_deployment(deployment_id)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="smoke-results.json")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    args = parser.parse_args()

    modalities = [item.strip() for item in os.environ.get("MODALITIES", "image,audio,video").split(",") if item.strip()]
    unknown = [item for item in modalities if item not in MODALITY_CONFIGS]
    if unknown:
        raise SystemExit(f"Unsupported modalities: {', '.join(unknown)}")

    if args.cleanup_only:
        _cleanup_stale_smoke_deployments()
        output_path = Path(args.output)
        output_path.write_text(json.dumps({"results": [], "failures": []}, indent=2), encoding="utf-8")
        return 0

    if not args.skip_cleanup:
        _cleanup_stale_smoke_deployments()

    preparation = _prepare_models(modalities)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for modality in modalities:
        try:
            results.append(_deployment_and_job(modality))
        except Exception as exc:
            failure = {
                "modality": modality,
                "error": str(exc),
                "type": type(exc).__name__,
            }
            failures.append(failure)
            results.append(failure)

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps({"preparation": preparation, "results": results, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    print(output_path.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())