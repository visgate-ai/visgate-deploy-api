#!/usr/bin/env python3
"""Run live image/audio/video smoke tests against the hosted Deploy API."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
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

WEBHOOK_EVENTS: dict[str, list[dict[str, Any]]] = {}
WEBHOOK_LOCK = threading.Lock()


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, method=method, data=data, headers=_auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
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


class _WebhookHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {"raw": raw.decode("utf-8", errors="replace")}
        event = {
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": body,
            "received_at": time.time(),
        }
        with WEBHOOK_LOCK:
            WEBHOOK_EVENTS.setdefault(self.path, []).append(event)
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def _start_webhook_server() -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _WebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _start_cloudflared(port: int) -> tuple[subprocess.Popen[str], str]:
    log_path = Path("cloudflared.log")
    if log_path.exists():
        log_path.unlink()
    process = subprocess.Popen(
        [
            "cloudflared",
            "tunnel",
            "--url",
            f"http://127.0.0.1:{port}",
            "--no-autoupdate",
            "--logfile",
            str(log_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 60
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            match = pattern.search(text)
            if match:
                return process, match.group(0)
        if process.poll() is not None:
            raise RuntimeError("cloudflared exited before publishing a tunnel URL")
        time.sleep(1)
    raise TimeoutError("Timed out waiting for cloudflared tunnel URL")


def _wait_for_tunnel_ready(tunnel_url: str, *, timeout_seconds: int = 90) -> None:
    deadline = time.time() + timeout_seconds
    probe_url = f"{tunnel_url.rstrip('/')}/healthz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe_url, timeout=10) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for tunnel readiness at {probe_url}")


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


def _wait_for_webhook(path: str, *, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with WEBHOOK_LOCK:
            events = list(WEBHOOK_EVENTS.get(path, []))
        if events:
            return events[-1]
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for webhook {path}")


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


def _deployment_and_job(modality: str, tunnel_base: str) -> dict[str, Any]:
    config = MODALITY_CONFIGS[modality]
    deployment_webhook_path = f"/webhook/deployment-{modality}"
    job_webhook_path = f"/webhook/job-{modality}"
    deployment_payload = dict(config["deployment"])
    deployment_payload["hf_token"] = HF_TOKEN
    deployment_payload["user_runpod_key"] = RUNPOD_API_KEY
    deployment_payload["user_webhook_url"] = f"{tunnel_base}{deployment_webhook_path}"

    deployment_response = _request_json("POST", f"{API_BASE}/v1/deployments", deployment_payload)
    deployment_id = deployment_response["deployment_id"]
    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    status_capture = _start_sse_capture(f"{API_BASE}/v1/deployments/{deployment_id}/stream")
    logs_capture = _start_sse_capture(f"{API_BASE}/v1/deployments/{deployment_id}/logs/stream")
    deployment_final = _poll_json(
        f"{API_BASE}/v1/deployments/{deployment_id}",
        {"ready", "failed", "webhook_failed", "deleted"},
        timeout_seconds=1500 if modality == "video" else 900,
    )
    status_events = _stop_sse_capture(status_capture)
    log_events = _stop_sse_capture(logs_capture)
    deployment_webhook = _wait_for_webhook(deployment_webhook_path, timeout_seconds=120)

    result: dict[str, Any] = {
        "modality": modality,
        "deployment_id": deployment_id,
        "deployment": deployment_final,
        "deployment_status_events": status_events,
        "deployment_log_events": log_events,
        "deployment_webhook": deployment_webhook,
        "live_log_tunnel_seen": _cloud_log_seen(deployment_id, started_at_iso),
    }

    if deployment_final["status"] != "ready":
        raise RuntimeError(f"{modality} deployment did not become ready: {deployment_final}")

    job_payload = dict(config["job"])
    job_payload["deployment_id"] = deployment_id
    job_payload["user_webhook_url"] = f"{tunnel_base}{job_webhook_path}"
    job_response = _request_json("POST", f"{API_BASE}/v1/inference/jobs", job_payload)
    job_id = job_response["job_id"]
    job_final = _poll_json(
        f"{API_BASE}/v1/inference/jobs/{job_id}",
        {"completed", "failed", "cancelled", "expired"},
        timeout_seconds=1800 if modality == "video" else 900,
    )
    job_webhook = _wait_for_webhook(job_webhook_path, timeout_seconds=180)

    result["job_id"] = job_id
    result["job"] = job_final
    result["job_webhook"] = job_webhook

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
        artifact = job_webhook.get("body", {}).get("artifact") or job_final.get("artifact")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="smoke-results.json")
    args = parser.parse_args()

    modalities = [item.strip() for item in os.environ.get("MODALITIES", "image,audio,video").split(",") if item.strip()]
    unknown = [item for item in modalities if item not in MODALITY_CONFIGS]
    if unknown:
        raise SystemExit(f"Unsupported modalities: {', '.join(unknown)}")

    server, port = _start_webhook_server()
    tunnel_process = None
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        tunnel_process, tunnel_url = _start_cloudflared(port)
        _wait_for_tunnel_ready(tunnel_url)
        for modality in modalities:
            try:
                results.append(_deployment_and_job(modality, tunnel_url))
            except Exception as exc:
                failure = {
                    "modality": modality,
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
                failures.append(failure)
                results.append(failure)
    finally:
        server.shutdown()
        server.server_close()
        if tunnel_process and tunnel_process.poll() is None:
            tunnel_process.terminate()
            try:
                tunnel_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel_process.kill()

    output_path = Path(args.output)
    output_path.write_text(json.dumps({"results": results, "failures": failures}, indent=2), encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())