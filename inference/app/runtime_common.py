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


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
    import json

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if VISGATE_INTERNAL_SECRET:
        req_headers["X-Visgate-Internal-Secret"] = VISGATE_INTERNAL_SECRET
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
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


def _worker_r2_credentials() -> tuple[str, str, str]:
    return (
        os.environ.get("VISGATE_R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID", ""),
        os.environ.get("VISGATE_R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        os.environ.get("VISGATE_R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL", ""),
    )


def download_r2_artifact_to_tempfile(artifact: dict[str, Any], suffix: str) -> str:
    bucket_name = artifact.get("bucket_name")
    object_key = artifact.get("key")
    if not bucket_name or not object_key:
        raise ValueError("R2 input artifact is missing bucket_name or key")
    access_key, secret_key, endpoint_url = _worker_r2_credentials()
    if not access_key or not secret_key or not endpoint_url:
        raise ValueError("Worker R2 credentials are not configured")

    tmp_path = ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = access_key
        env["AWS_SECRET_ACCESS_KEY"] = secret_key
        env["AWS_REGION"] = "auto"
        subprocess.run(
            [
                "s5cmd",
                "--endpoint-url",
                endpoint_url,
                "cp",
                f"s3://{bucket_name}/{object_key}",
                tmp_path,
            ],
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return tmp_path
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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


def _upload_with_boto3(
    tmp_path: str,
    s3_config: dict[str, Any],
    bucket_name: str,
    object_key: str,
    content_type: str,
) -> None:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=s3_config.get("endpointUrl"),
        aws_access_key_id=s3_config.get("accessId", ""),
        aws_secret_access_key=s3_config.get("accessSecret", ""),
        region_name="auto",
    )
    client.upload_file(tmp_path, bucket_name, object_key, ExtraArgs={"ContentType": content_type})


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
        try:
            subprocess.run(["s5cmd", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            log_tunnel("WARNING", f"s5cmd upload failed, retrying with boto3: {exc}")
            print(f"s5cmd upload failed, retrying with boto3: {exc}")
            if not bucket_name or not object_key or not s3_config:
                raise
            _upload_with_boto3(tmp_path, s3_config, bucket_name, object_key, content_type)

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
    except Exception as exc:
        log_tunnel("ERROR", f"Artifact upload failed: {exc}")
        print(f"Artifact upload failed: {exc}")
        return None
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def download_to_tempfile(url: str, suffix: str) -> str:
    max_bytes = int(os.getenv("MAX_INPUT_DOWNLOAD_BYTES", str(512 * 1024 * 1024)))
    chunk_size = 1024 * 1024
    tmp_path = ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        written = 0
        with urllib.request.urlopen(url, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("Input file exceeds MAX_INPUT_DOWNLOAD_BYTES")
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError("Input file exceeds MAX_INPUT_DOWNLOAD_BYTES")
                    out.write(chunk)
        return tmp_path
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise