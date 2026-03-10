"""Shared helpers for modality-specific RunPod workers."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import urllib.request
import uuid
from typing import Any, Optional

from app.config import (
    CDN_BASE_URL,
    DEFAULT_OUTPUT_KEY_PREFIX,
    RUNPOD_API_KEY,
    VISGATE_DEPLOYMENT_ID,
    VISGATE_INTERNAL_SECRET,
    VISGATE_LOG_TUNNEL,
    VISGATE_WEBHOOK,
)


def mask_sensitive(text: str) -> str:
    if not text:
        return text
    for token in ("hf_", "rpa_", "sk_", "api_key", "token", "secret"):
        if token in text.lower():
            return "***REDACTED***"
    return text


def post_json(url: str, payload: dict[str, Any]) -> None:
    import json

    headers = {"Content-Type": "application/json"}
    if VISGATE_INTERNAL_SECRET:
        headers["X-Visgate-Internal-Secret"] = VISGATE_INTERNAL_SECRET
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30):
        return


def log_tunnel(level: str, message: str) -> None:
    if not VISGATE_LOG_TUNNEL or not VISGATE_LOG_TUNNEL.strip():
        return
    try:
        post_json(VISGATE_LOG_TUNNEL, {"level": level, "message": mask_sensitive(message)})
    except Exception:
        return


def request_cleanup(reason: str) -> None:
    if not VISGATE_DEPLOYMENT_ID or not VISGATE_WEBHOOK:
        return
    cleanup_url = VISGATE_WEBHOOK.replace("/deployment-ready/", "/cleanup/")
    payload: dict[str, Any] = {"reason": reason}
    if RUNPOD_API_KEY:
        payload["runpod_api_key"] = RUNPOD_API_KEY
    try:
        post_json(cleanup_url, payload)
    except Exception:
        return


def job_s3_config(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any] | None:
    raw = job.get("s3Config") or job_input.get("s3Config")
    if not isinstance(raw, dict) or not raw:
        return None
    return raw


def artifact_target(
    s3_config: dict[str, Any] | None,
    *,
    extension: str,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    if not s3_config:
        return None, None, None, None, None
    bucket_name = s3_config.get("bucketName")
    endpoint_url = s3_config.get("endpointUrl")
    if not bucket_name or not endpoint_url:
        return None, None, None, None, None
    key_prefix = (s3_config.get("keyPrefix") or DEFAULT_OUTPUT_KEY_PREFIX).strip("/")
    object_key = f"{key_prefix}/{int(time.time())}_{uuid.uuid4().hex}.{extension.lstrip('.')}"
    upload_target = f"s3://{bucket_name}/{object_key}"
    object_url = f"{endpoint_url.rstrip('/')}/{bucket_name}/{object_key}"
    return upload_target, bucket_name, endpoint_url, object_key, object_url


def upload_bytes(
    data: bytes,
    job: dict[str, Any],
    job_input: dict[str, Any],
    *,
    content_type: str,
    extension: str,
) -> Optional[dict[str, Any]]:
    s3_config = job_s3_config(job, job_input)
    upload_target, bucket_name, endpoint_url, object_key, object_url = artifact_target(
        s3_config,
        extension=extension,
    )
    if not upload_target:
        return None
    try:
        res = subprocess.run(["s5cmd", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"s5cmd version check: {res.stdout.decode().strip()}")
    except Exception as e:
        print(f"s5cmd search failed: {e}")
        return None

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{extension.lstrip('.')}", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        env = os.environ.copy()
        cmd = ["s5cmd"]
        if s3_config:
            env["AWS_ACCESS_KEY_ID"] = s3_config.get("accessId", "")
            env["AWS_SECRET_ACCESS_KEY"] = s3_config.get("accessSecret", "")
            env["AWS_REGION"] = "auto"  # recommended for R2
            if s3_config.get("endpointUrl"):
                cmd.extend(["--endpoint-url", s3_config["endpointUrl"]])
        cmd.extend(["cp", tmp_path, upload_target])
        print(f"Executing: {' '.join(cmd)} to {upload_target}")
        res = subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"s5cmd success: {res.stdout.decode().strip()}")
        url = object_url or upload_target
        if CDN_BASE_URL and object_key:
            url = f"{CDN_BASE_URL.rstrip('/')}/{object_key}"
        return {
            "bucket_name": bucket_name,
            "endpoint_url": endpoint_url,
            "key": object_key,
            "url": url,
            "content_type": content_type,
            "bytes": len(data),
        }
    except Exception:
        return None
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def download_to_tempfile(url: str, suffix: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name